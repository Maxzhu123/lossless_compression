#!/usr/bin/env bash
# Run from baselines/ with the project's CUDA Python environment activated.
set -e

python -m pip install setuptools safetensors
cuda_major=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')
python -m pip install "nvidia-nvcomp-cu${cuda_major}"
python -c 'import triton' || python -m pip install triton
python _vendor/zipnn/build.py build_ext --inplace

# Build and cache DFloat11's native encoder.
cd ..
python -c 'from baselines._dfloat11_encoder import _library; _library()'
