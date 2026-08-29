#!/usr/bin/env python3
"""Training script for bridging experiments.

This script trains sparse adapters (e.g., transcoder) with bridging losses
that encourage layer-wise compatibility with a reference model.
"""

import os
import sys
from helpers import logger, setup_logging
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse
import wandb

from training.config import load_config, ExperimentConfig, BridgingConfig, _finalize_config
from training.dataset import OpenThoughtsDataset, collate_fn
from training.forward_utils import forward_mixed, sample_cutoffs
from training.losses import compute_kl_loss, compute_lm_loss, compute_nmse_loss
from models import get_transcoder_classes

DEBUG_MODE_EARLY_EXIT_STEPS = 50

def move_batch_to(device, batch):
    """Move batch tensors to device."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def setup_models_bridging(config: ExperimentConfig):
    """Load transcoder model and frozen reference model (bridging mode).

    Model assembly:
      1. Load Qwen2ForCausalLMWithTranscoder via from_pretrained. This loads all
         standard Qwen2 weights (attention, embeddings, MLP gate/up/down) from the
         checkpoint. Transcoder parameters (transcoder_enc, transcoder_dec) are NOT
         in the checkpoint and stay at their __init__ values (dec=zeros, so zero
         initial contribution).
      2. For "target" backbone: the checkpoint is the reference model (DeepSeek R1
         Distill), so we additionally swap in the base model's MLP weights
         (gate_proj, up_proj, down_proj). This gives us: reference attention/embed
         + base MLP + fresh transcoder.
      3. After training, model.save_pretrained() saves everything (including trained
         transcoder weights) as a single checkpoint — no conversion step needed.
    """
    bridging_config = config.bridging
    assert bridging_config is not None
    backbone = getattr(bridging_config, 'backbone', 'base')
    tc_config = config.transcoder
    assert tc_config is not None

    assert config.model_arch is not None, "model_arch must be set (auto-detected or explicit)"
    ConfigWithTranscoder, ModelWithTranscoder = get_transcoder_classes(config.model_arch)

    # Load tokenizer from reference model (the model we're distilling toward)
    tokenizer_path = bridging_config.reference_model_path if bridging_config else config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build HF config with transcoder params
    hf_config = ConfigWithTranscoder.from_pretrained(
        config.model_name,
        transcoder_n_features=tc_config.n_features,
        transcoder_dec_bias=tc_config.dec_bias,
    )

    # Load transcoder model. from_pretrained loads standard weights from the
    # checkpoint; transcoder_enc/dec are not in the checkpoint and stay at __init__
    # values (dec=zeros → zero initial contribution).
    if backbone == "target":
        # Target backbone: load reference model (attn/embed/layernorm from reference),
        # then swap in base model's MLP weights. Result: reference attn + base MLP + fresh transcoder.
        logger.info(f"Loading reference model as backbone: {bridging_config.reference_model_path}")
        model = ModelWithTranscoder.from_pretrained(
            bridging_config.reference_model_path,
            config=hf_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"Swapping in base model MLP weights from: {config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        for adapter_mlp, base_layer in zip(model._transcoder_mlps(), base_model.model.layers): # pyright: ignore[reportCallIssue]
            base_mlp = base_layer.mlp  # type: ignore[union-attr]
            device = adapter_mlp.gate_proj.weight.device
            adapter_mlp.gate_proj.weight.data.copy_(base_mlp.gate_proj.weight.data.to(device))
            adapter_mlp.up_proj.weight.data.copy_(base_mlp.up_proj.weight.data.to(device))
            adapter_mlp.down_proj.weight.data.copy_(base_mlp.down_proj.weight.data.to(device))
        del base_model
        logger.info("MLP weights swapped")
    else:
        # Base backbone: all non-transcoder weights from base model directly.
        logger.info(f"Loading base model: {config.model_name}")
        model = ModelWithTranscoder.from_pretrained(
            config.model_name,
            config=hf_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Re-initialize transcoder weights. from_pretrained with device_map="auto" creates
    # meta tensors first, so our __init__ zero-initialization of dec is overwritten by
    # HF's default _init_weights (normal distribution). This restores dec=zeros for
    # zero initial transcoder contribution.
    for mlp in model._transcoder_mlps(): # pyright: ignore[reportCallIssue]
        mlp._init_transcoder_weights()

    # Freeze everything except transcoder parameters
    for name, param in model.named_parameters():
        param.requires_grad = "transcoder" in name
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Trainable: {n_trainable:,} params, Frozen: {n_frozen:,} params")

    # Load reference model (frozen)
    logger.info(f"Loading reference model: {bridging_config.reference_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        bridging_config.reference_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Reference model loaded and frozen")

    # Copy config values from target model so vLLM behaves correctly at eval time
    model.config.max_position_embeddings = ref_model.config.max_position_embeddings
    model.config.eos_token_id = ref_model.config.eos_token_id

    return model, ref_model, tokenizer



def setup_models_resume(config: ExperimentConfig, resume_from: str):
    """Load model and tokenizer from a checkpoint for resuming training."""
    tokenizer = AutoTokenizer.from_pretrained(resume_from, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Resuming model from: {resume_from}")
    model = Qwen2ForCausalLMWithTranscoder.from_pretrained(
        resume_from,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze everything except transcoder parameters
    for name, param in model.named_parameters():
        param.requires_grad = "transcoder" in name
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable:,} params")

    # Load reference model if bridging mode
    ref_model = None
    if config.bridging:
        print(f"Loading reference model: {config.bridging.reference_model_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.bridging.reference_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    return model, ref_model, tokenizer


def setup_data(config: ExperimentConfig, tokenizer) -> PredefinedDataset:
    """Setup dataset loader. Dataloaders are created per-epoch via get_epoch_dataloaders()."""

    dataset_loader = PredefinedDataset(
        dataset_entries=config.datasets,
        tokenizer=tokenizer,
        loss_on_prompt=config.loss_on_prompt,
        batch_size=config.batch_size,
        total_rows=config.total_rows,
        weight_by=config.weight_by,
    )

    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=collate_with_tokenizer,
        num_workers=0,
        generator=generator
    )

    # Optional validation dataset
    val_dataloader = None
    if hasattr(config, 'val_data_path') and config.val_data_path:
        print(f"Loading validation dataset from: {config.val_data_path}")
        val_dataset = OpenThoughtsDataset(
            data_path=config.val_data_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            format=config.data_format,
            truncate=config.truncate,
            loss_on_prompt=config.loss_on_prompt,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_with_tokenizer,
            num_workers=0
        )

    return train_dataset, train_dataloader, val_dataloader, generator


def setup_training(config: ExperimentConfig, model, dataset_loader: PredefinedDataset):
    """Setup optimizer and scheduler."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=0.0
    )

    assert dataset_loader._loaded_datasets is not None, "Datasets must be loaded before setting up training"
    train_size = len(dataset_loader._loaded_datasets["train"])
    steps_per_epoch = train_size // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler, total_steps, warmup_steps, train_size



def train_step_bridging(
    model,
    ref_model,
    batch: dict,
    config: ExperimentConfig,
    global_step: int,
    total_steps: int,
    gradient_accumulation_steps: int,
) -> dict[str, float]:
    """Single training step with bridging loss using memory-efficient forward_mixed.

    This function handles its own backward() calls to free computation graphs
    immediately after each loss component, avoiding memory buildup.

    Returns:
        Dict of metrics (losses are already backward'd, no tensor returned)
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    bridging_config = config.bridging
    assert bridging_config is not None
    assert config.transcoder is not None
    n_layers = len(model.model.layers)
    l1_weight = config.transcoder.l1_weight or 0.0

    # 1. Main forward pass with feature caching (for L1 loss + stats)
    model.set_cache_features(True)
    logits_adapt = model(input_ids=input_ids, attention_mask=attention_mask).logits
    model.set_cache_features(False)

    # Collect sparsity loss (sum of L1 across layers, must be while graph alive)
    raw_sparsity = model.collect_sparsity_loss()
    sparsity_loss = l1_weight * raw_sparsity

    # 2. Clean ref forward (for logits_ref if using KL loss)
    with torch.no_grad():
        logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Track metrics
    metrics = {}
    total_loss_value = 0.0

    # 3. Adapt-only loss + sparsity loss (backward together)
    # Compute the training loss with grad, and the other one without grad for logging
    adapt_only_loss = torch.tensor(0.0, device=logits_adapt.device)
    if bridging_config.loss_type == "kl":
        kl_to_ref = compute_kl_loss(logits_adapt, logits_ref, labels)
        with torch.no_grad():
            lm_loss = compute_lm_loss(logits_adapt, labels)
        if bridging_config.always_include_adapt_only:
            adapt_only_loss = kl_to_ref
    else:
        lm_loss = compute_lm_loss(logits_adapt, labels)
        with torch.no_grad():
            kl_to_ref = compute_kl_loss(logits_adapt, logits_ref, labels)
        if bridging_config.always_include_adapt_only:
            adapt_only_loss = lm_loss

    metrics["train/kl_to_ref"] = kl_to_ref.item()
    metrics["train/lm_loss"] = lm_loss.item()
    if bridging_config.always_include_adapt_only:
        metrics["bridging/adapt_only"] = adapt_only_loss.item()
        total_loss_value += bridging_config.lambda_adapt * adapt_only_loss.item()

    # Combined loss for first backward (adapt_only + sparsity)
    first_loss = bridging_config.lambda_adapt * adapt_only_loss + sparsity_loss
    if first_loss.requires_grad:
        scaled_first = first_loss / gradient_accumulation_steps
        scaled_first.backward()

    metrics["train/sparsity"] = raw_sparsity.item()

    # Free logits_adapt to save memory
    del logits_adapt

    # 4. NMSE loss (if enabled) - backward separately
    if bridging_config.lambda_nmse > 0:
        nmse_loss, nmse_layerwise = compute_nmse_loss(
            model, ref_model, input_ids, attention_mask, return_layerwise=True
        )
        scaled_nmse = (bridging_config.lambda_nmse * nmse_loss) / gradient_accumulation_steps
        scaled_nmse.backward()
        metrics["bridging/nmse"] = nmse_loss.item()
        total_loss_value += bridging_config.lambda_nmse * nmse_loss.item()
        # Log per-layer NMSE
        for layer_idx, nmse_val in nmse_layerwise.items(): # type: ignore
            metrics[f"bridging_layerwise/nmse_layer_{layer_idx}"] = nmse_val
        del nmse_loss

    # 5. Bridging losses at sampled cutoffs (each backward immediately)
    bridge_loss_total = 0.0
    if bridging_config.n_cutoffs > 0:
        cutoffs = sample_cutoffs(n_layers, bridging_config.n_cutoffs, bridging_config.sampling)

        for k in cutoffs:
            # adapt->ref: adapter layers 0..k-1, ref layers k..L
            # Skip k=0 (no adapter layers used, no gradients)
            if k > 0:
                logits_a2r = forward_mixed(model, ref_model, input_ids, attention_mask, switch_layer=k)
                # Compute training loss with grad, other for logging only
                if bridging_config.loss_type == "kl":
                    loss_a2r = compute_kl_loss(logits_a2r, logits_ref, labels)
                    with torch.no_grad():
                        loss_a2r_lm = compute_lm_loss(logits_a2r, labels)
                    metrics[f"bridging_layerwise/a2r_kl_layer_{k}"] = loss_a2r.item()
                    metrics[f"bridging_layerwise/a2r_lm_layer_{k}"] = loss_a2r_lm.item()
                else:
                    loss_a2r = compute_lm_loss(logits_a2r, labels)
                    with torch.no_grad():
                        loss_a2r_kl = compute_kl_loss(logits_a2r, logits_ref, labels)
                    metrics[f"bridging_layerwise/a2r_lm_layer_{k}"] = loss_a2r.item()
                    metrics[f"bridging_layerwise/a2r_kl_layer_{k}"] = loss_a2r_kl.item()
                scaled_a2r = (bridging_config.lambda_bridge * loss_a2r) / (len(cutoffs) * gradient_accumulation_steps)
                scaled_a2r.backward()
                bridge_loss_total += loss_a2r.item()
                del logits_a2r, loss_a2r

            # ref->adapt: ref layers 0..k-1, adapter layers k..L
            # Skip k=n_layers (no adapter layers used, no gradients)
            if k < n_layers:
                logits_r2a = forward_mixed(ref_model, model, input_ids, attention_mask, switch_layer=k)
                # Compute training loss with grad, other for logging only
                if bridging_config.loss_type == "kl":
                    loss_r2a = compute_kl_loss(logits_r2a, logits_ref, labels)
                    with torch.no_grad():
                        loss_r2a_lm = compute_lm_loss(logits_r2a, labels)
                    metrics[f"bridging_layerwise/r2a_kl_layer_{k}"] = loss_r2a.item()
                    metrics[f"bridging_layerwise/r2a_lm_layer_{k}"] = loss_r2a_lm.item()
                else:
                    loss_r2a = compute_lm_loss(logits_r2a, labels)
                    with torch.no_grad():
                        loss_r2a_kl = compute_kl_loss(logits_r2a, logits_ref, labels)
                    metrics[f"bridging_layerwise/r2a_lm_layer_{k}"] = loss_r2a.item()
                    metrics[f"bridging_layerwise/r2a_kl_layer_{k}"] = loss_r2a_kl.item()
                scaled_r2a = (bridging_config.lambda_bridge * loss_r2a) / (len(cutoffs) * gradient_accumulation_steps)
                scaled_r2a.backward()
                bridge_loss_total += loss_r2a.item()
                del logits_r2a, loss_r2a

        # Average bridge loss for logging
        metrics["bridging/bridge"] = bridge_loss_total / (2 * len(cutoffs))
        total_loss_value += bridging_config.lambda_bridge * metrics["bridging/bridge"]

    metrics["bridging/total"] = total_loss_value
    metrics["train/total_loss"] = total_loss_value

    return metrics


def train_epoch(
    model,
    ref_model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    config: ExperimentConfig,
    epoch: int,
    starting_step: int,
    total_steps: int,
    total_samples_seen: int,
    val_dataloader=None,
    skip_batches: int = 0,
    dl_generator=None,
):
    """Train for one epoch."""
    model.train()
    batch_losses = []
    global_step = starting_step
    accumulation_step = 0
    current_batch_losses = []
    current_metrics = {}
    samples_seen = total_samples_seen

    micro_batch_size = config.micro_batch_size or config.batch_size

    gradient_accumulation_steps = (
        config.batch_size // micro_batch_size
    ) # Note: not rly doing gradient accumulation anymore, see PredefinedDataset._make_dataloader comment.

    embed_device = model.get_input_embeddings().weight.device

    # Capture generator state BEFORE iter() consumes it for shuffling.
    # This is saved in checkpoints so resume can reproduce this epoch's shuffle order.
    epoch_rng_state = dl_generator.get_state().clone() if dl_generator is not None else None

    dataloader_iter = iter(dataloader)
    if skip_batches > 0:
        print(f"  Skipping {skip_batches} batches to resume position...")
        for _ in tqdm(range(skip_batches), desc="Skipping batches"):
            next(dataloader_iter)
        print(f"  Done, resuming training")

    remaining = len(dataloader) - skip_batches
    epoch_pbar = tqdm(dataloader_iter, total=remaining, desc=f"Epoch {epoch+1}/{config.num_epochs}")

    for step, batch in enumerate(epoch_pbar):
        batch = move_batch_to(embed_device, batch)

        # Forward + backward (handled inside train_step for memory efficiency)
        step_metrics = train_step_bridging(
            model, ref_model, batch, config,
            global_step, total_steps,
            gradient_accumulation_steps
        )

        accumulation_step += 1
        current_batch_losses.append(step_metrics.get("train/total_loss", 0.0))
        samples_seen += micro_batch_size

        # Accumulate metrics
        for k, v in step_metrics.items():
            if k not in current_metrics:
                current_metrics[k] = []
            current_metrics[k].append(v)

        # Accumulate truncation stats from batch
        if "truncated" in batch:
            if "data/truncated" not in current_metrics:
                current_metrics["data/truncated"] = []
                current_metrics["data/original_length"] = []
            current_metrics["data/truncated"].extend(batch["truncated"])
            current_metrics["data/original_length"].extend(batch["original_length"])

        # Optimizer step
        if accumulation_step % gradient_accumulation_steps == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip_norm
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Average losses for this batch
            avg_batch_loss = sum(current_batch_losses) / len(current_batch_losses)
            batch_losses.append(avg_batch_loss)

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            epoch_pbar.set_postfix({
                'loss': f"{avg_batch_loss:.4f}",
                'lr': f"{current_lr:.2e}",
                'grad_norm': f"{total_norm:.3f}",
            })

            # Collect transcoder stats for logging
            for stat_name, stat_value in model.collect_transcoder_stats().items():
                current_metrics[stat_name] = [stat_value]

            # Log to WandB
            if config.use_wandb:
                log_dict = {
                    "train/total_loss": avg_batch_loss,
                    "train/learning_rate": current_lr,
                    "train/gradient_norm": total_norm.item(),
                    "train/epoch": epoch,
                    "train/step": global_step,
                    "train/samples_seen": samples_seen,
                }
                # Add averaged metrics (includes data/truncated as fraction, data/original_length as avg)
                for k, values in current_metrics.items():
                    log_dict[k] = sum(values) / len(values) if values else 0.0

                wandb.log(log_dict, step=global_step)

            # Reset accumulators
            current_batch_losses = []
            current_metrics = {}

            model.clear_cached_stats()

            # Run validation
            if val_dataloader is not None and global_step % config.val_frequency == 0:
                val_metrics = validate_bridging(model, ref_model, val_dataloader, config)
                val_metrics["val/epoch"] = epoch
                val_metrics["val/step"] = global_step
                val_metrics["val/samples_seen"] = samples_seen
                print(f"  Val total: {val_metrics['val/total_loss']:.4f}, LM: {val_metrics['val/language_modeling_loss']:.4f}, KL: {val_metrics['val/kl_to_ref']:.4f}")
                if config.use_wandb:
                    wandb.log(val_metrics, step=global_step)

            # Run comprehensive layerwise validation
            if val_dataloader is not None and global_step % config.layerwise_val_frequency == 0:
                print(f"  Running layerwise validation...")
                layerwise_metrics = validate_layerwise(model, ref_model, val_dataloader, config)
                if config.use_wandb:
                    wandb.log(layerwise_metrics, step=global_step)

            # Save periodic checkpoint (overwrites previous latest)
            if config.save_checkpoints and global_step > 0 and global_step % config.checkpoint_frequency == 0:
                print(f"  Saving checkpoint at step {global_step}...")
                training_state = {
                    'step': global_step,
                    'epoch': epoch,
                    'samples_seen': samples_seen,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'dataloader_rng': epoch_rng_state,
                }
                save_latest_checkpoint(model, tokenizer, config.output_dir, global_step,
                                       training_state=training_state)

            # Debug mode early exit
            if config.debug_mode and global_step >= DEBUG_MODE_EARLY_EXIT_STEPS:
                logger.info(f"Debug mode: Breaking after {global_step} steps")
                break

        del batch

    avg_epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
    return avg_epoch_loss, global_step, samples_seen



@torch.no_grad()
def validate_bridging(
    model,
    ref_model,
    val_dataloader,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Run validation on 100 samples (bridging mode)."""
    assert config.bridging is not None
    model.eval()

    total_lm_loss = 0.0
    total_kl_loss = 0.0
    total_sparsity_loss = 0.0
    total_loss = 0.0
    num_samples = 0
    max_samples = 100

    embed_device = model.get_input_embeddings().weight.device

    for batch in val_dataloader:
        if num_samples >= max_samples:
            break

        batch = move_batch_to(embed_device, batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward with feature caching for sparsity
        model.set_cache_features(True)
        logits_adapt = model(input_ids=input_ids, attention_mask=attention_mask).logits
        model.set_cache_features(False)
        raw_sparsity = model.collect_sparsity_loss().item()
        model.clear_cached_stats()
        total_sparsity_loss += raw_sparsity

        logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Compute LM loss
        lm_loss = compute_lm_loss(logits_adapt, labels)
        total_lm_loss += lm_loss.item()

        # Compute KL loss (adapter -> ref)
        kl_loss = compute_kl_loss(logits_adapt, logits_ref, labels)
        total_kl_loss += kl_loss.item()

        # Total loss (using whatever loss_type is configured)
        if config.bridging.loss_type == "kl":
            task_loss = kl_loss.item()
        else:
            task_loss = lm_loss.item()
        total_loss += task_loss

        num_samples += 1
        del batch

    model.train()

    return {
        "val/total_loss": total_loss / num_samples if num_samples > 0 else 0.0,
        "val/language_modeling_loss": total_lm_loss / num_samples if num_samples > 0 else 0.0,
        "val/kl_to_ref": total_kl_loss / num_samples if num_samples > 0 else 0.0,
        "val/sparsity": total_sparsity_loss / num_samples if num_samples > 0 else 0.0,
    }


@torch.no_grad()
def validate_layerwise(
    model,
    ref_model,
    val_dataloader,
    config: ExperimentConfig,
    max_samples: int = 25,
) -> dict[str, float]:
    """Compute comprehensive layer-wise bridging and NMSE metrics on a small sample."""
    model.eval()

    n_layers = len(model.model.layers)
    embed_device = model.get_input_embeddings().weight.device

    # Accumulators for layer-wise metrics
    a2r_lm = {k: 0.0 for k in range(1, n_layers + 1)}
    a2r_kl = {k: 0.0 for k in range(1, n_layers + 1)}
    r2a_lm = {k: 0.0 for k in range(n_layers)}
    r2a_kl = {k: 0.0 for k in range(n_layers)}
    nmse_by_layer = {k: 0.0 for k in range(n_layers + 1)}
    num_samples = 0

    for batch in val_dataloader:
        if num_samples >= max_samples:
            break

        batch = move_batch_to(embed_device, batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Get ref logits for KL computation
        logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Layer-wise bridging losses
        for k in range(n_layers + 1):
            if k > 0:  # a2r: adapter 0..k-1, ref k..L
                logits_a2r = forward_mixed(model, ref_model, input_ids, attention_mask, switch_layer=k)
                a2r_lm[k] += compute_lm_loss(logits_a2r, labels).item()
                a2r_kl[k] += compute_kl_loss(logits_a2r, logits_ref, labels).item()
                del logits_a2r

            if k < n_layers:  # r2a: ref 0..k-1, adapter k..L
                logits_r2a = forward_mixed(ref_model, model, input_ids, attention_mask, switch_layer=k)
                r2a_lm[k] += compute_lm_loss(logits_r2a, labels).item()
                r2a_kl[k] += compute_kl_loss(logits_r2a, logits_ref, labels).item()
                del logits_r2a

        # NMSE by layer
        _, nmse_layerwise = compute_nmse_loss(model, ref_model, input_ids, attention_mask, return_layerwise=True)
        assert isinstance(nmse_layerwise, dict)
        for layer_idx, val in nmse_layerwise.items():
            nmse_by_layer[layer_idx] += val

        num_samples += 1
        del batch, logits_ref

    model.train()

    # Build results
    results = {}
    for k in range(1, n_layers + 1):
        results[f"val_layerwise/a2r_lm_layer_{k}"] = a2r_lm[k] / num_samples
        results[f"val_layerwise/a2r_kl_layer_{k}"] = a2r_kl[k] / num_samples
    for k in range(n_layers):
        results[f"val_layerwise/r2a_lm_layer_{k}"] = r2a_lm[k] / num_samples
        results[f"val_layerwise/r2a_kl_layer_{k}"] = r2a_kl[k] / num_samples
    for k in range(n_layers + 1):
        results[f"val_layerwise/nmse_layer_{k}"] = nmse_by_layer[k] / num_samples

    return results


def _load_token_metrics_all_examples(config: ExperimentConfig, tokenizer) -> list | None:
    """Load all available token metrics evaluation examples.

    Auto-selects the data source based on the reference model: evalchemy_qwen for
    Qwen/DeepSeek reference models, lmsys_chat for others.

    Returns None if data cannot be loaded (logs a warning in that case).
    """
    from analysis.evals.compute_token_metrics_onpolicy import (
        EVALCHEMY_DIR, load_all_rollouts, load_lmsys_examples,
    )

    ref_path = config.bridging.reference_model_path if config.bridging else ""
    is_qwen_ref = any(k in ref_path.lower() for k in ("qwen", "deepseek"))

    try:
        if is_qwen_ref:
            eval_dir = EVALCHEMY_DIR / "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B"
            return load_all_rollouts(str(eval_dir))
        else:
            return load_lmsys_examples(tokenizer)
    except Exception as e:
        logger.warning(f"Could not load token metrics examples: {e}. Token metrics eval will be skipped.")
        return None


def _run_and_log_token_metrics(
    model,
    ref_model,
    tokenizer,
    examples: list,
    config: ExperimentConfig,
    global_step: int,
):
    """Run token metrics evaluation and log all results to wandb.

    Temporarily sets the model to eval mode, runs KL divergence and top-1 agreement
    evaluation, then restores the original training mode.

    Returns the EvalResults, or None if the eval failed.
    """
    from analysis.evals.compute_token_metrics_onpolicy import run_token_metrics_eval

    logger.info(f"Running token metrics eval on {len(examples)} examples...")
    was_training = model.training
    model.eval()

    try:
        results = run_token_metrics_eval(
            eval_model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            examples=examples,
            save_output=False,
        )
    finally:
        if was_training:
            model.train()
        torch.cuda.empty_cache()

    if not config.use_wandb or wandb.run is None:
        return results

    log_dict: dict[str, float] = {
        "token_metrics/kl_mean": results.kl_mean,
        "token_metrics/top1_agreement": results.top1_agreement,
        "token_metrics/kl_mean_interesting": results.kl_mean_interesting,
        "token_metrics/top1_agreement_interesting": results.top1_agreement_interesting,
        "token_metrics/n_tokens": float(results.n_tokens),
        "token_metrics/n_interesting": float(results.n_interesting),
    }
    for bm, bm_metrics in results.per_benchmark.items():
        log_dict[f"token_metrics/{bm}/kl_mean"] = bm_metrics.kl_mean
        log_dict[f"token_metrics/{bm}/top1_agreement"] = bm_metrics.top1_agreement
        log_dict[f"token_metrics/{bm}/kl_mean_interesting"] = bm_metrics.kl_mean_interesting
        log_dict[f"token_metrics/{bm}/top1_agreement_interesting"] = bm_metrics.top1_agreement_interesting

    wandb.log(log_dict, step=global_step)
    return results


def save_checkpoint(model, tokenizer, config: ExperimentConfig, output_dir):
    """Save full model checkpoint (no conversion needed)."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_config(config, os.path.join(output_dir, CHECKPOINT_CONFIG_FILENAME))
    logger.info(f"Checkpoint saved to {output_dir}")


def save_latest_checkpoint(model, tokenizer, base_dir, step, training_state=None):
    """Save checkpoint to base_dir/latest_step_N, removing any previous latest_step_* dir."""
    import glob
    import shutil
    output_dir = os.path.join(base_dir, f"latest_step_{step}")
    save_checkpoint(model, tokenizer, output_dir)
    if training_state is not None:
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
        print(f"  Training state saved (step={step})")
    # Delete old checkpoints after new one is fully written
    for old_dir in glob.glob(os.path.join(base_dir, "latest_step_*")):
        if old_dir != output_dir:
            shutil.rmtree(old_dir)


def main():
    parser = argparse.ArgumentParser(description="Train with bridging loss")
    parser.add_argument("--config", required=True, nargs="+", help="Path(s) to experiment config YAMLs. Later files shallow-override earlier ones.")
    parser.add_argument("--learning_rate", "-lr", type=float, help="Override learning rate")
    parser.add_argument("--l1_weight", type=float, help="Override L1 weight")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    setup_logging()

    if args.sweep and args.sweep_id:
        parser.error("--sweep and --sweep_id are mutually exclusive.")
    elif args.sweep or args.sweep_id:
        _run_sweep(args, parser)
    else:
        _run_training(args, parser=parser)


def _run_sweep(args, parser: argparse.ArgumentParser | None = None):
    """Create or join a wandb sweep and launch an agent that calls _run_training for each run."""
    overrides = _build_overrides(args, parser)
    base_config = load_config(args.config, overrides=overrides)

    if args.sweep_id:
        sweep_id = args.sweep_id
        logger.info(f"Joining existing sweep {sweep_id}")
    else:
        import yaml
        with open(args.sweep, 'r') as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep=sweep_config, project=base_config.wandb_project)
        logger.info(f"Created sweep {sweep_id}")

    logger.info(f"Starting agent with {args.sweep_count} runs")

    def sweep_train():
        _run_training(args, parser=parser, sweep_mode=True)

    wandb.agent(sweep_id, function=sweep_train, count=args.sweep_count, project=base_config.wandb_project)


def _build_overrides(args, parser: argparse.ArgumentParser | None = None) -> dict[str, Any]:
    """Build overrides dict from CLI args (all non-None args except config/sweep args)."""
    exclude = {"config", "sweep", "sweep_id", "sweep_count"}
    overrides: dict[str, Any] = {
        k: v for k, v in vars(args).items()
        if k not in exclude and v is not None
    }

    # debug_mode comes in as a string from argparse, convert to bool
    if "debug_mode" in overrides:
        if overrides["debug_mode"].lower() == "true":
            overrides["debug_mode"] = True
        elif overrides["debug_mode"].lower() == "false":
            overrides["debug_mode"] = False
        else:
            msg = f"Invalid value for --debug_mode: '{overrides['debug_mode']}'. Must be 'true' or 'false'."
            if parser:
                parser.error(msg)
            raise ValueError(msg)

    return overrides


def _run_training(args, parser: argparse.ArgumentParser | None = None, sweep_mode: bool = False):
    """Run a single training session. Called directly or by wandb.agent."""
    overrides = _build_overrides(args, parser)

    # Load config with CLI overrides
    config = load_config(args.config, overrides=overrides)

    # Sweeps require wandb
    if sweep_mode and not config.use_wandb:
        raise ValueError("Cannot run sweeps with use_wandb=False (e.g. debug_mode). Remove --debug_mode or set use_wandb: true.")

    # Verify Hub access before training so we fail fast
    if config.push_to_hub:
        from training.upload_models.hub import build_hub_repo_id, verify_hub_access

        hub_repo_id = build_hub_repo_id(config)
        verify_hub_access(hub_repo_id)
    else:
        hub_repo_id = None

    # WandB — initialize early so setup logs are captured
    if sweep_mode:
        wandb.init()
        # Apply sweep parameters on top of config
        for key, value in dict(wandb.config).items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Sweep override {key}: {value}")
        # Regenerate computed fields since sweep changed hyperparams
        config.wandb_run_name = None
        config.output_dir = None
        from training.config import _finalize_config
        config = _finalize_config(config)
        wandb.config.update(config.__dict__, allow_val_change=True)
    elif config.use_wandb:
        wandb.init(project=config.wandb_project, name=f"bridging_{config.wandb_run_name}", config=config.__dict__)

    if config.bridging is None:
        raise ValueError("Config must specify a 'bridging' section.")

    # Apply overrides
    config_changed = False

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
        print(f"Override learning rate: {args.learning_rate}")
        config_changed = True

    if args.l1_weight is not None and config.transcoder:
        config.transcoder.l1_weight = args.l1_weight
        print(f"Override L1 weight: {args.l1_weight}")
        config_changed = True

    if config_changed:
        # Update run name and output dir to reflect the overrides
        config.wandb_run_name = None  # Force regeneration
        config.output_dir = None      # Force regeneration
        config = _finalize_config(config)  # Regenerate names with new params
        print(f"Updated run name: {config.wandb_run_name}")
        print(f"Updated output dir: {config.output_dir}")

    print(f"Starting bridging training")
    print(f"  Base model: {config.model_name}")
    print(f"  Reference model: {config.bridging.reference_model_path}")
    print(f"  Loss type: {config.bridging.loss_type}")
    print(f"  N cutoffs: {config.bridging.n_cutoffs}")

    # Setup models
    if args.resume_from:
        model, ref_model, tokenizer = setup_models_resume(config, args.resume_from)
    else:
        model, ref_model, tokenizer = setup_models_bridging(config)

    train_dataset, train_dataloader, val_dataloader, dl_generator = setup_data(config, tokenizer)
    optimizer, scheduler, total_steps, warmup_steps = setup_training(config, model, train_dataset)

    # Resume: restore optimizer and scheduler from checkpoint
    resume_step = 0
    resume_epoch = 0
    resume_samples = 0
    if args.resume_from:
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"No training_state.pt found in {args.resume_from}. "
                "Cannot resume without saved training state."
            )
        print(f"Loading training state from: {state_path}")
        training_state = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(training_state['optimizer'])
        # Move optimizer states to match parameter devices (handles different GPU layouts)
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                state = optimizer.state[param]
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(param.device)
        scheduler.load_state_dict(training_state['scheduler'])
        resume_step = training_state['step']
        resume_epoch = training_state['epoch']
        resume_samples = training_state['samples_seen']
        if 'dataloader_rng' in training_state:
            dl_generator.set_state(training_state['dataloader_rng'])
        print(f"  Resumed from step={resume_step}, epoch={resume_epoch}, samples={resume_samples}")

    # WandB
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=f"bridging_{config.wandb_run_name}",
            config=config.__dict__
        )

    print(f"Training setup:")
    print(f"  - Train dataset size: {len(train_dataset)}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Warmup steps: {warmup_steps}")
    if args.resume_from:
        print(f"  - Resuming from step: {resume_step}")

    # Training loop
    current_step = resume_step
    total_samples_seen = resume_samples
    steps_per_epoch = len(train_dataset) // config.batch_size
    grad_accum = config.batch_size // config.micro_batch_size

    for epoch in range(resume_epoch, config.num_epochs):
        # On the resume epoch, skip batches we already trained on
        skip = 0
        if epoch == resume_epoch and resume_step > 0:
            steps_into_epoch = resume_step - resume_epoch * steps_per_epoch
            skip = steps_into_epoch * grad_accum

        epoch_loss, current_step, total_samples_seen = train_epoch(
            model, ref_model, tokenizer, train_dataloader, optimizer, scheduler,
            config,
            epoch, current_step, total_steps, total_samples_seen,
            val_dataloader=val_dataloader,
            skip_batches=skip,
            dl_generator=dl_generator,
        )

        # Save checkpoint at end of epoch (overwrites previous latest)
        if config.save_checkpoints:
            training_state = {
                'step': current_step,
                'epoch': epoch + 1,
                'samples_seen': total_samples_seen,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'dataloader_rng': dl_generator.get_state(),
            }
            save_latest_checkpoint(model, tokenizer, config.output_dir, current_step,
                                   training_state=training_state)

    logger.info("Training complete!")

    # Final post-training token metrics eval
    if final_token_examples is not None and ref_model is not None:
        logger.info("Running final post-training token metrics eval...")
        final_results = _run_and_log_token_metrics(
            model, ref_model, tokenizer, final_token_examples, config, current_step,
        )
        if final_results is not None and config.use_wandb and wandb.run is not None:
            wandb.run.summary["token_metrics_final/kl_mean"] = final_results.kl_mean
            wandb.run.summary["token_metrics_final/top1_agreement"] = final_results.top1_agreement
            wandb.run.summary["token_metrics_final/kl_mean_interesting"] = final_results.kl_mean_interesting
            wandb.run.summary["token_metrics_final/top1_agreement_interesting"] = final_results.top1_agreement_interesting
            for bm, bm_metrics in final_results.per_benchmark.items():
                wandb.run.summary[f"token_metrics_final/{bm}/kl_mean"] = bm_metrics.kl_mean
                wandb.run.summary[f"token_metrics_final/{bm}/top1_agreement"] = bm_metrics.top1_agreement
                wandb.run.summary[f"token_metrics_final/{bm}/kl_mean_interesting"] = bm_metrics.kl_mean_interesting
                wandb.run.summary[f"token_metrics_final/{bm}/top1_agreement_interesting"] = bm_metrics.top1_agreement_interesting

    # Always save final checkpoint
    save_checkpoint(model, tokenizer, config, config.output_dir)

    # Push to Hugging Face Hub (hub_repo_id computed and verified before training)
    if hub_repo_id:
        from training.upload_models.hub import push_to_hub

        logger.info(f"Pushing model to Hub: {hub_repo_id}")
        wandb_url = wandb.run.url if (config.use_wandb and wandb.run is not None) else None
        try:
            push_to_hub(model, tokenizer, config, hub_repo_id, wandb_url=wandb_url)
        except Exception as e:
            logger.exception(f"Failed to push model to Hub: {e}", exc_info=True, stack_info=True)

        if config.use_wandb and wandb.run is not None:
            wandb.run.summary["hf_model_url"] = f"https://huggingface.co/{hub_repo_id}"
            wandb.run.summary["hf_repo_id"] = hub_repo_id

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
