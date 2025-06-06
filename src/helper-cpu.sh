#!/bin/bash

# Please change -m options before running the automated job
# 1. CPU
# 2. GPU
# 3. CUDA
# 4. JAX
# 5. JIT
# 6. DASK

(
  python controller.py -n 104 -m 1 &
  wait
  python notification.py
) &