"""Optional wandb logging for feature-activation collection runs.

Full-corpus collections (e.g. the entire ~100k-row lmsys val split) run for many
hours, so this logs live throughput (sequences/s, tokens/s), progress, and a final
summary to a wandb project, making long runs monitorable from anywhere. Shared by
the base and adapter collectors. It is a complete no-op unless ``--wandb`` is set, so
smoke runs never import or touch wandb.

CLI: ``add_collection_wandb_args(parser)`` adds the flags; build a
``CollectionWandbLogger`` from the parsed args with ``from_args``.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from helpers.log import logger

DEFAULT_WANDB_PROJECT = "transcoder-feature-collection"


def add_collection_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--wandb*`` flags to a collector's argument parser."""
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log live throughput/progress and a final summary to Weights & Biases. Off by default; "
        "enable for long full-corpus runs you want to monitor. Requires being logged into wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help=f"wandb project for the collection run (default: {DEFAULT_WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="wandb entity/team (default: your wandb default entity).",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="wandb run name (default: the output directory name).",
    )
    parser.add_argument(
        "--wandb_log_every",
        type=int,
        default=1,
        help="Log throughput every N batches (default: 1). Raise to reduce wandb traffic on fast runs.",
    )


class CollectionWandbLogger:
    """Lightweight throughput/progress logger. No-ops entirely when disabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        total_sequences: int | None = None,
        log_every: int = 1,
    ) -> None:
        self.enabled = enabled
        self._total = total_sequences
        self._log_every = max(1, int(log_every))
        self._seqs = 0
        self._tokens = 0
        self._batches = 0
        self._t0 = time.time()
        if not enabled:
            return
        import wandb  # imported lazily so disabled runs never need the dependency

        self._wandb = wandb
        wandb.init(
            project=project or DEFAULT_WANDB_PROJECT,
            entity=entity,
            name=run_name,
            config=config or {},
        )
        logger.info(
            f"wandb logging enabled: project={project or DEFAULT_WANDB_PROJECT} run={run_name} "
            f"url={getattr(wandb.run, 'url', '?')}"
        )

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        run_name: str,
        config: dict[str, Any],
        total_sequences: int | None,
    ) -> "CollectionWandbLogger":
        return cls(
            enabled=bool(getattr(args, "wandb", False)),
            project=getattr(args, "wandb_project", None),
            entity=getattr(args, "wandb_entity", None),
            run_name=getattr(args, "wandb_run_name", None) or run_name,
            config=config,
            total_sequences=total_sequences,
            log_every=getattr(args, "wandb_log_every", 1),
        )

    def reset_timer(self) -> None:
        """Start the throughput clock (call right before the batch loop)."""
        self._t0 = time.time()

    def log_batch(self, *, n_seqs: int, n_tokens: int) -> None:
        """Accumulate one processed batch and log throughput every ``log_every`` batches."""
        self._seqs += n_seqs
        self._tokens += n_tokens
        self._batches += 1
        if not self.enabled or (self._batches % self._log_every):
            return
        elapsed = max(1e-6, time.time() - self._t0)
        metrics = {
            "progress/sequences_done": self._seqs,
            "progress/tokens_done": self._tokens,
            "progress/batches_done": self._batches,
            "throughput/sequences_per_s": self._seqs / elapsed,
            "throughput/tokens_per_s": self._tokens / elapsed,
            "throughput/elapsed_s": elapsed,
        }
        if self._total:
            metrics["progress/fraction"] = self._seqs / self._total
        self._wandb.log(metrics)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        """Write summary scalars and close the run."""
        if not self.enabled:
            return
        elapsed = max(1e-6, time.time() - self._t0)
        run_summary = {
            "final/sequences": self._seqs,
            "final/tokens": self._tokens,
            "final/batches": self._batches,
            "final/elapsed_s": elapsed,
            "final/sequences_per_s": self._seqs / elapsed,
            "final/tokens_per_s": self._tokens / elapsed,
            **(summary or {}),
        }
        for key, value in run_summary.items():
            self._wandb.run.summary[key] = value
        self._wandb.finish()
