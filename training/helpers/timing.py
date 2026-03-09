import time

from training.helpers.log import logger


def fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as e.g. '5s' or '2m05s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


class Timer:
    """Context manager that prints elapsed time for a labeled block.

    Usage:
        with Timer("load dataset"):
            ds = load_dataset(...)
        # prints: [load dataset: 1m30s]
    """

    def __init__(self, label: str):
        self.label = label
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        logger.info(f"  [{self.label}: {fmt_elapsed(time.time() - self.start)}]")
