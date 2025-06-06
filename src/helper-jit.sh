#!/bin/bash

# Please change -m options before running the automated job
# 1. CPU
# 2. GPU
# 3. CUDA
# 4. JAX
# 5. JIT
# 6. DASK

(
  for i in {0..7}
  do
    python controller.py -n 13 -i $i -m 5 &
  done
  wait
  python notification.py
) &