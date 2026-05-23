"""Upload circuit-tracer feature caches to Hugging Face Hub."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

from helpers.log import logger
from training.upload_models.hub import truncate_repo_name, verify_hub_access


FEATURE_REPO_PREFIX = "2026.TA.features"
FEATURE_REPO_NAME_MAX_LEN = 96
FEATURE_CONFIG_FILENAME = "feature_collection_config.json"
FEATURE_SUMMARY_FILENAME = "feature_collection_summary.json"

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.rstrip("/").split("/")[-1] or value).strip("_")
    return slug or "run"


def _dataset_id_from_val_data(entry: str) -> str | None:
    if entry.startswith("hf://"):
        source = entry
    elif ":" in entry:
        source = entry.split(":", 1)[1]
    else:
        source = entry
    if source.startswith("hf://"):
        parts = source[len("hf://"):].split("/", 2)
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    if source.startswith("/"):
        return None
    parts = source.split("/", 2)
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return source or None


def dataset_ids_from_val_data(val_data: list[str]) -> list[str]:
    ids: list[str] = []
    for entry in val_data:
        dataset_id = _dataset_id_from_val_data(entry)
        if dataset_id and dataset_id not in ids:
            ids.append(dataset_id)
    return ids


def feature_collection_config_from_args(args: Any) -> dict[str, Any]:
    """Build the duplicate-detection config for a feature collection run."""
    payload = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "output_dir",
            "hub_org",
            "hf_feature_repo_id",
            "upload_circuit_tracer_features_to_hub",
        }
    }
    payload["val_data"] = list(payload.get("val_data") or [])
    payload["upload_format"] = "circuit_tracer_packed_features_v1"
    payload["features_path_in_repo"] = "features/"
    return payload


def feature_collection_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def build_feature_collection_repo_id(args: Any) -> tuple[str, dict[str, Any]]:
    """Return the deterministic Hub repo ID and config for this collection run."""
    config = feature_collection_config_from_args(args)
    if getattr(args, "hf_feature_repo_id", None):
        return args.hf_feature_repo_id, config

    org = getattr(args, "hub_org", None)
    if not org:
        org = HfApi().whoami()["name"]
    fingerprint = feature_collection_fingerprint(config)
    val_data_hash = hashlib.sha256(
        json.dumps(config["val_data"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    name = "_".join(
        [
            FEATURE_REPO_PREFIX,
            _slugify(str(config["model_path"])),
            f"ms{config.get('max_samples', 'all')}",
            f"ml{config.get('max_length')}",
            f"tk{config.get('top_k')}",
            f"dtk{config.get('domain_top_k')}",
            f"rk{config.get('n_random')}",
            f"ctx{config.get('context_before')}-{config.get('context_after')}",
            f"data{val_data_hash}",
            f"h{fingerprint}",
        ]
    )
    return f"{org}/{truncate_repo_name(name, max_len=FEATURE_REPO_NAME_MAX_LEN)}", config


def _repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        return True
    except RepositoryNotFoundError:
        return False


def reserve_feature_collection_repo(repo_id: str, config: dict[str, Any]) -> bool:
    """Verify access and reserve the repo.

    Returns True when the repo already exists, which means another run has
    already reserved or completed the same deterministic feature collection.
    """
    verify_hub_access(repo_id)
    api = HfApi()
    if _repo_exists(api, repo_id):
        existing_config = _download_existing_config(repo_id)
        if existing_config is not None and existing_config != config:
            raise RuntimeError(
                f"Hugging Face feature repo already exists with different collection config: "
                f"https://huggingface.co/{repo_id}. Choose a different --hf_feature_repo_id "
                "or remove the stale repo."
            )
        logger.info(
            "Found existing Hugging Face feature repo for this collection: "
            f"https://huggingface.co/{repo_id}"
        )
        return True

    logger.info(f"Reserving Hugging Face feature repo: {repo_id}")
    api.create_repo(repo_id, repo_type="model", exist_ok=False)
    _upload_json(api, repo_id, FEATURE_CONFIG_FILENAME, config, "Reserve feature collection repo")
    _push_feature_model_card(
        repo_id=repo_id,
        config=config,
        status="Reserved; feature collection is in progress.",
    )
    return False


def _download_existing_config(repo_id: str) -> dict[str, Any] | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=FEATURE_CONFIG_FILENAME,
            repo_type="model",
        )
    except Exception:
        return None
    return json.loads(Path(path).read_text())


def upload_circuit_tracer_features_to_hub(
    *,
    repo_id: str,
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """Upload packed circuit-tracer feature files and metadata to Hub."""
    api = HfApi()
    packed_dir = output_dir / "circuit_tracer_features"
    if not (packed_dir / "index.json.gz").exists():
        raise FileNotFoundError(
            f"Cannot upload circuit-tracer features; missing {packed_dir / 'index.json.gz'}"
        )

    logger.info(f"Uploading packed circuit-tracer features to https://huggingface.co/{repo_id}")
    api.upload_folder(
        folder_path=str(packed_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="features",
        commit_message="Upload circuit-tracer feature cache",
    )
    _upload_json(api, repo_id, FEATURE_CONFIG_FILENAME, config, "Update feature collection config")

    summary = _build_feature_summary(output_dir, config)
    _upload_json(api, repo_id, FEATURE_SUMMARY_FILENAME, summary, "Add feature collection summary")
    _push_feature_model_card(
        repo_id=repo_id,
        config=config,
        status="Complete.",
        summary=summary,
    )
    logger.info(f"Uploaded circuit-tracer features: https://huggingface.co/{repo_id}")


def _upload_json(api: HfApi, repo_id: str, path_in_repo: str, payload: dict[str, Any], message: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / path_in_repo
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=message,
        )


def _build_feature_summary(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = output_dir / "feature_metadata.json"
    summary: dict[str, Any] = {
        "model_path": config.get("model_path"),
        "val_data": config.get("val_data", []),
        "features_path_in_repo": "features/",
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        summary.update(
            {
                "total_tokens": metadata.get("total_tokens"),
                "tokens_per_domain": metadata.get("tokens_per_domain"),
                "tokens_per_region": metadata.get("tokens_per_region"),
                "feature_frequency_summary": metadata.get("feature_frequency_summary"),
            }
        )
    return summary


def _push_feature_model_card(
    *,
    repo_id: str,
    config: dict[str, Any],
    status: str,
    summary: dict[str, Any] | None = None,
) -> None:
    datasets = dataset_ids_from_val_data(list(config.get("val_data") or []))
    card_data = ModelCardData(
        base_model=str(config.get("model_path")),
        datasets=datasets,
        tags=["transcoder-adapters", "circuit-tracer", "feature-cache"],
        library_name="circuit-tracer",
    )
    lines = [
        f"# {repo_id.split('/')[-1]}",
        "",
        "Packed circuit-tracer feature cache produced by `analysis.features.collect_feature_activations`.",
        "",
        "## Status",
        "",
        status,
        "",
        "## Circuit-Tracer Usage",
        "",
        f"Use this repo ID as the scan name: `{repo_id}`.",
        "The frontend reads packed features from `features/index.json.gz` and `features/layer_N.bin`.",
        "",
        "## Collection Inputs",
        "",
        f"- **Model**: [{config.get('model_path')}](https://huggingface.co/{config.get('model_path')})",
        f"- **Config**: [{FEATURE_CONFIG_FILENAME}]({FEATURE_CONFIG_FILENAME})",
    ]
    if datasets:
        lines.append("- **Datasets**:")
        lines.extend(f"  - [{dataset}](https://huggingface.co/datasets/{dataset})" for dataset in datasets)
    if summary:
        lines.extend([
            "",
            "## Summary",
            "",
            f"- **Total tokens**: {summary.get('total_tokens')}",
            f"- **Tokens per domain**: `{summary.get('tokens_per_domain')}`",
        ])

    content = "\n".join(lines) + "\n"
    ModelCard(content=f"---\n{card_data.to_yaml()}\n---\n{content}").push_to_hub(repo_id)


def load_circuit_tracer_feature_from_hub(repo_id: str, feature_index: int) -> dict[str, Any]:
    """Load one packed feature from the Hub using circuit-tracer's expected layout."""
    layer_idx, feature_idx = _cantor_unpair(feature_index)
    index_path = Path(
        hf_hub_download(repo_id=repo_id, filename="features/index.json.gz", repo_type="model")
    )
    with gzip.open(index_path, "rt") as f:
        index = json.load(f)
    layer_entry = index[str(layer_idx)]
    offsets = layer_entry["offsets"]
    start = int(offsets[feature_idx])
    end = int(offsets[feature_idx + 1])
    bin_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"features/{layer_entry['filename']}",
            repo_type="model",
        )
    )
    raw = bin_path.read_bytes()[start:end]
    compressed_len = struct.unpack("<I", raw[:4])[0]
    return json.loads(gzip.decompress(raw[4:4 + compressed_len]).decode("utf-8"))


def _cantor_unpair(z: int) -> tuple[int, int]:
    w = int(((8 * z + 1) ** 0.5 - 1) / 2)
    t = w * (w + 1) // 2
    y = z - t
    x = w - y
    return x, y
