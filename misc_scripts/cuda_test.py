from __future__ import annotations

import torch

from helpers.log import logger, setup_logging


def main() -> int:
    setup_logging()
    logger.info("Starting CUDA test")
    logger.info("Torch version: %s", torch.__version__)

    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return 1

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    logger.info("CUDA is available on device 0: %s", device_name)

    tensor = torch.ones((1024, 1024), device=device)
    result = tensor.sum().item()
    torch.cuda.synchronize(device)

    allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    logger.info("Allocated tensor on GPU with sum=%s", result)
    logger.info("CUDA memory allocated after tensor allocation: %.2f MiB", allocated_mb)
    logger.info("CUDA test completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
