# Theoretical TFLOPS ≠ Real-world Performance
# Testing Theoretical Maximum FLOPS on GPUs

This project aims to measure the theoretical maximum FLOPS (Floating Point Operations Per Second) achievable on various GPU models. Please see the original work by [Stas Bekman](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-flops).

## Key Features

1. **Optimized Search**: Unlike the [original implementation](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py) which uses a brute force approach, this version leverages Optuna for efficient parameter optimization.

2. **Visualization**: Optuna provides insightful visualizations of the optimization process:

   ![Optuna Optimization Visualization](./img/optuna1.png)
   ![Optuna Optimization Visualization](./img/optuna2.png)

3. **Data Collection**: An optional feature allows submitting results to a remote API for data collection and analysis.

## Acknowledgements

Special thanks to [Stas Bekman](https://x.com/StasBekman) for the original implementation and research.

# Stats

| GPU Model | Best Shape (MxNxK) | TFLOPS |
|-----------|---------------------|--------|
| NVIDIA RTX 4090 | 15360x6784x8192 | 178.5 |
| NVIDIA A100 PCIe | 2304x5120x1536 | 256.4 |
| NVIDIA A100 SXM | 6912x16384x2048 | 267.9 |
| NVIDIA H100 PCIe | 6912x16384x2048 | 499.5 |
| AMD MI300X | 4352x13568x3840 | 758.3 |
| NVIDIA H100 SXM | 6144x17920x2816 | 792.1 |


# Install

```
# If you are not using uv, install it: pip install uv
git clone https://github.com/mag-/gpu_benchmark
cd gpu_benchmark
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
./mamf-finder.py
```

# Changes:
- added optuna for finding the parameters instead of doing a brute-force search


# TODO:
- check raw CUDA
- check tinygrad

