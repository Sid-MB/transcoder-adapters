"""Export integrated transcoder adapters to circuit-tracer transcoder-set format.

This converts checkpoints with per-layer ``mlp.transcoder_enc`` /
``mlp.transcoder_dec`` modules into the per-layer safetensors layout expected by
circuit-tracer's ``transcoder_set`` loader.

Example:
    uv run python -m analysis.attribution.export_circuit_tracer_transcoders \
        siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 \
        --model_name google/gemma-2-2b \
        --output_dir /tmp/gemma2_tc8192_circuit_tracer

Then, you can use it with:

    uv run --extra viz circuit-tracer attribute \
        --model google/gemma-2-2b \
        --transcoder_set /tmp/gemma2_tc8192_circuit_tracer \
        --prompt "The capital of France is" \
        --graph_output_path /tmp/gemma2_tc8192_circuit_tracer/france_capital.pt \
        --dtype bfloat16

And visualize with:

    uv run --extra viz circuit-tracer attribute \
        --model google/gemma-2-2b \
        --transcoder_set /tmp/gemma2_tc8192_circuit_tracer \
        --prompt "The capital of France is" \
        --slug france_capital \
        --graph_file_dir /tmp/gemma2_tc8192_circuit_tracer/graphs \
        --server \
        --port 8041 \
        --dtype bfloat16

    # If this is running on a remote machine, forward the port and open:
    #   http://localhost:8041

    
uv run --extra viz circuit-tracer attribute \
        --model google/gemma-2-2b \
        --transcoder_set $LARGE_ARTIFACTS_DIR/transcoder-adapters/circuit_tracer_transcoders/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260520_233833_15506755 \
        --prompt "The capital of France is" \
        --graph_output_path $LARGE_ARTIFACTS_DIR/transcoder-adapters/circuit_tracer_transcoders/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260520_233833_15506755/france_capital.pt \
        --dtype bfloat16

        uv run --extra viz circuit-tracer attribute \
        --model google/gemma-2-2b \
        --transcoder_set $LARGE_ARTIFACTS_DIR/transcoder-adapters/circuit_tracer_transcoders/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260520_233833_15506755 \
        --prompt "The capital of France is" \
        --slug france_capital \
        --graph_file_dir $LARGE_ARTIFACTS_DIR/transcoder-adapters/circuit_tracer_transcoders/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260520_233833_15506755/graphs \
        --server \
        --port 8041 \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from helpers.log import logger, setup_logging


_ENC_WEIGHT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.transcoder_enc\.weight$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class CheckpointTensorSource:
    """Lazy tensor reader for local or Hugging Face sharded safetensors checkpoints."""

    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self.local_dir = Path(checkpoint) if Path(checkpoint).exists() else None
        self._weight_map: dict[str, str] | None = None
        self._single_file: str | None = None
        self._tensor_names: list[str] | None = None

    @property
    def is_local(self) -> bool:
        return self.local_dir is not None

    def metadata_path(self, filename: str) -> Path:
        if self.local_dir is not None:
            return self.local_dir / filename
        return Path(hf_hub_download(repo_id=self.checkpoint, filename=filename))

    def _load_weight_map(self) -> dict[str, str]:
        if self._weight_map is not None:
            return self._weight_map

        index_path = self._try_metadata_path("model.safetensors.index.json")
        if index_path is not None:
            with index_path.open("r") as f:
                payload = json.load(f)
            self._weight_map = dict(payload["weight_map"])
            return self._weight_map

        single_path = self._try_metadata_path("model.safetensors")
        if single_path is None:
            raise FileNotFoundError(
                f"Could not find model.safetensors.index.json or model.safetensors in {self.checkpoint!r}"
            )

        self._single_file = "model.safetensors"
        with safe_open(single_path, framework="pt", device="cpu") as f:
            self._weight_map = {key: self._single_file for key in f.keys()}
        return self._weight_map

    def _try_metadata_path(self, filename: str) -> Path | None:
        if self.local_dir is not None:
            path = self.local_dir / filename
            return path if path.exists() else None
        try:
            return Path(hf_hub_download(repo_id=self.checkpoint, filename=filename))
        except Exception:
            return None

    def model_config(self) -> dict[str, Any]:
        config_path = self.metadata_path("config.json")
        with config_path.open("r") as f:
            return json.load(f)

    def training_model_name(self) -> str | None:
        path = self._try_metadata_path("training-config.yaml")
        if path is None:
            return None
        for line in path.read_text().splitlines():
            if line.startswith("model_name:"):
                return line.split(":", 1)[1].strip().strip("'\"") or None
        return None

    def tensor_names(self) -> list[str]:
        if self._tensor_names is None:
            self._tensor_names = sorted(self._load_weight_map())
        return self._tensor_names

    def tensor(self, name: str) -> torch.Tensor:
        weight_map = self._load_weight_map()
        try:
            filename = weight_map[name]
        except KeyError as exc:
            raise KeyError(f"Tensor {name!r} not found in checkpoint {self.checkpoint!r}") from exc

        if self.local_dir is not None:
            shard_path = self.local_dir / filename
        else:
            shard_path = Path(hf_hub_download(repo_id=self.checkpoint, filename=filename))

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def discover_layers(source: CheckpointTensorSource) -> list[int]:
    layers = []
    for name in source.tensor_names():
        match = _ENC_WEIGHT_RE.match(name)
        if match is not None:
            layers.append(int(match.group(1)))

    if not layers:
        raise ValueError(
            f"No tensors matching model.layers.N.mlp.transcoder_enc.weight in {source.checkpoint!r}"
        )

    layers = sorted(layers)
    expected = list(range(layers[-1] + 1))
    if layers != expected:
        raise ValueError(f"Expected contiguous layers {expected}, found {layers}")
    return layers


def export_layer(source: CheckpointTensorSource, layer: int, output_dir: Path) -> None:
    prefix = f"model.layers.{layer}.mlp"
    w_enc = source.tensor(f"{prefix}.transcoder_enc.weight").contiguous()
    b_enc = source.tensor(f"{prefix}.transcoder_enc.bias").contiguous()
    w_dec_hf = source.tensor(f"{prefix}.transcoder_dec.weight")
    w_dec = w_dec_hf.T.contiguous()

    b_dec_name = f"{prefix}.transcoder_dec.bias"
    if b_dec_name in source.tensor_names():
        b_dec = source.tensor(b_dec_name).contiguous()
    else:
        b_dec = torch.zeros(w_dec.shape[1], dtype=w_dec.dtype)

    if w_enc.shape[0] != b_enc.shape[0]:
        raise ValueError(f"Layer {layer}: encoder weight/bias feature dimensions do not match")
    if w_dec.shape[0] != w_enc.shape[0]:
        raise ValueError(f"Layer {layer}: encoder and decoder feature dimensions do not match")
    if w_dec.shape[1] != w_enc.shape[1]:
        raise ValueError(f"Layer {layer}: encoder and decoder model dimensions do not match")
    if b_dec.shape[0] != w_dec.shape[1]:
        raise ValueError(f"Layer {layer}: decoder bias and decoder output dimensions do not match")

    save_file(
        {
            "W_enc": w_enc,
            "W_dec": w_dec,
            "b_enc": b_enc,
            "b_dec": b_dec,
        },
        output_dir / f"layer_{layer}.safetensors",
    )


def write_config(
    output_dir: Path,
    *,
    model_name: str,
    feature_input_hook: str,
    feature_output_hook: str,
    activation: str,
) -> None:
    config = {
        "model_name": model_name,
        "model_kind": "transcoder_set",
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "activation": activation,
    }
    with (output_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def default_output_dir(checkpoint: str) -> Path:
    from helpers.paths.output_path import generate_output_path

    checkpoint_name = Path(checkpoint.rstrip("/")).name or "checkpoint"
    slug = _SLUG_RE.sub("_", checkpoint_name).strip("_") or "checkpoint"
    return generate_output_path("circuit_tracer_transcoders", slug, consistent=True)


def require_new_conversion_output_dir(output_dir: Path, *, conversion_name: str) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"{conversion_name} output directory already exists at {output_dir}. "
            "We've already done the conversion; remove the directory or choose a different "
            "--output_dir to run it again."
        )


def export_circuit_tracer_transcoders(
    checkpoint: str,
    output_dir: Path,
    *,
    model_name: str | None = None,
    feature_input_hook: str = "ln2.hook_normalized",
    feature_output_hook: str = "hook_mlp_out",
    activation: str = "relu",
) -> None:
    require_new_conversion_output_dir(
        output_dir,
        conversion_name="Circuit-tracer transcoder",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source = CheckpointTensorSource(checkpoint)
    hf_config = source.model_config()
    resolved_model_name = model_name or source.training_model_name() or hf_config.get("_name_or_path")
    if not resolved_model_name:
        raise ValueError("Could not infer model_name. Pass --model_name explicitly.")

    layers = discover_layers(source)
    logger.info(f"Found {len(layers)} transcoder layers in {checkpoint}")
    logger.info(f"Writing circuit-tracer transcoder set to {output_dir}")

    for layer in layers:
        export_layer(source, layer, output_dir)
        logger.info(f"Exported layer {layer}")

    write_config(
        output_dir,
        model_name=resolved_model_name,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        activation=activation,
    )
    logger.info(f"Wrote config.yaml for model_name={resolved_model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a transcoder-adapted HF checkpoint to circuit-tracer transcoder_set format."
    )
    parser.add_argument("checkpoint", help="HF repo ID or local checkpoint directory")
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for config.yaml and layer_N.safetensors. "
            "Default: PRODUCTS_DIR/circuit_tracer_transcoders/<checkpoint>_<timestamp>_<jobid>"
        ),
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Base model name to write in config.yaml. This is the model which the transcoders attach to. Defaults to training-config.yaml model_name if present.",
    )
    parser.add_argument("--feature_input_hook", default="ln2.hook_normalized")
    parser.add_argument("--feature_output_hook", default="hook_mlp_out")
    parser.add_argument("--activation", default="relu", choices=["relu"])
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir(args.checkpoint)
    export_circuit_tracer_transcoders(
        args.checkpoint,
        output_dir,
        model_name=args.model_name,
        feature_input_hook=args.feature_input_hook,
        feature_output_hook=args.feature_output_hook,
        activation=args.activation,
    )


if __name__ == "__main__":
    main()
