# GPU Tests

This directory contains opt-in tests for CUDA-only training paths. They are kept
outside the default CPU test suite so normal local test runs do not require a
GPU.

Run on a GPU-enabled machine:

```sh
./tests/gpu/run_gpu_suite.sh
```

On a machine without CUDA, the runner exits successfully after printing a skip
message.
