# Project Instructions

## Logging

Do not use `print()` statements. Use the structured logger from `helpers/log.py`:

```python
from helpers.log import logger

logger.info("message")
logger.warning("message")
logger.error("message")
```

## Tools
Use `uv run python` for Python. Run all Python code requiring CUDA or PyTorch or a GPU outside of the sandbox!

## Debugging
Try to validate changes by running your programs with really simple and quick parameters or on a toy model / example. For programs requiring model loading or a gpu, if you're on a GPU, you can run directly with `uv run python -m`, otherwise you can use slurm to start a GPU job. For example, to test analysis/simple_load/simple_load.py, you can use the following script on the GPU:
```sh
uv run python -m analysis.simple_load.simple_load "siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860" --prompt="Hi"
```
Feel free to ask about how you should test something or what the appropriate parameters are to use.
