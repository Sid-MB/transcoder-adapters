import logging
import sys
import time
from contextlib import contextmanager
logger = logging.getLogger("training")


class _WallTimeFormatter(logging.Formatter):
    """Formatter that appends wall time from wandb start when a run is active."""

    def format(self, record):
        try:
            import wandb
            if wandb.run is not None:
                elapsed = int(time.time() - wandb.run.start_time)
                minutes, seconds = divmod(elapsed, 60)
                record.wall_time = f" | {minutes:02d}m {seconds:02d}s"
            else:
                record.wall_time = ""
        except Exception:
            record.wall_time = ""
        return super().format(record)


def setup_logging(level: int = logging.INFO):
    """Configure the training logger. Call once at program start."""
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_WallTimeFormatter("[%(asctime)s%(wall_time)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)


@contextmanager
def log_group(name: str):
    """Log a named group with start/end markers."""
    logger.info(f"--- {name} ---")
    try:
        yield
    finally:
        logger.info(f"--- {name} done ---")
