import logging
import sys
from contextlib import contextmanager

logger = logging.getLogger("training")


def setup_logging(level: int = logging.INFO):
    """Configure the training logger. Call once at program start."""
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)


@contextmanager
def log_group(name: str):
    """Log a named group with start/end markers."""
    logger.info(f"--- {name} ---")
    try:
        yield
    finally:
        logger.info(f"--- {name} done ---")
