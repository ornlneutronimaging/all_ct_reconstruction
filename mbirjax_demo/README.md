# MBIRJAX CUDA Deployment Tests

This directory contains verification tests for the mbirjax CUDA environment deployment.

## Purpose

These tests verify that:
1. mbirjax is correctly installed with CUDA 12 support
2. JAX can detect and use GPU devices
3. Actual CT reconstruction works on GPU
4. GPU performance is within expected range

## Files

| File | Description |
|------|-------------|
| `test_mbirjax_cuda.py` | Command-line test script for automated verification |
| `test_mbirjax.ipynb` | Interactive Jupyter notebook for manual verification |
| `README.md` | This file |

## Prerequisites

The tests require the `pixi_mbirjax.toml` environment to be installed:

```bash
# From the repository root
pixi install --manifest-path pixi_mbirjax.toml
```

## Running the Tests

### Option 1: Command-Line Script (Recommended for CI/CD)

```bash
# Using pixi
pixi run --manifest-path pixi_mbirjax.toml test-cuda

# Or directly
pixi run --manifest-path pixi_mbirjax.toml python tests/test_mbirjax_cuda.py
```

**Exit Codes:**
- `0` - All tests passed, CUDA backend working
- `1` - Import errors or missing dependencies
- `2` - CUDA/GPU not available
- `3` - Reconstruction test failed
- `4` - Performance test suggests GPU not being used

### Option 2: Jupyter Notebook (For Interactive Verification)

```bash
# Start JupyterLab
pixi run --manifest-path pixi_mbirjax.toml lab

# Then open tests/test_mbirjax.ipynb
```

Or execute the notebook non-interactively:

```bash
pixi run --manifest-path pixi_mbirjax.toml test-notebook
```

## What the Tests Verify

### 1. Package Imports
- `numpy` is available
- `jax` is available with correct version
- `mbirjax` is available

### 2. CUDA Device Detection
- JAX can detect GPU devices
- Reports number and type of available GPUs

### 3. JAX GPU Backend
- Confirms JAX can execute computations on GPU
- Verifies device placement

### 4. MBIRJAX Reconstruction
- Creates synthetic CT sinogram data
- Runs `mj.ParallelBeamModel` reconstruction
- Verifies output shape, data type, and values (no NaN/Inf)

### 5. GPU Performance
- Runs multiple reconstruction iterations
- Ensures performance is within expected GPU range
- Warning if reconstruction seems CPU-bound

## Expected Output (Success)

```
============================================================
  MBIRJAX CUDA Installation Verification
============================================================
Python version: 3.10.x

============================================================
  Test 1: Package Imports
============================================================
  [✓ PASS] Import mbirjax, jax, numpy
          numpy version: 1.24.x
          jax version: 0.4.x
          mbirjax version: 0.6.x

============================================================
  Test 2: CUDA Device Detection
============================================================
  [✓ PASS] CUDA/GPU devices available
          Total devices: 2
          GPU devices: 1
          CPU devices: 1
          GPU device info:
            GPU 0: cuda:0

============================================================
  Test 3: JAX GPU Backend
============================================================
  [✓ PASS] JAX GPU computation
          Computation device: cuda:0
          Device platform: gpu

============================================================
  Test 4: MBIRJAX Reconstruction
============================================================
  [✓ PASS] CT reconstruction on synthetic data
          Sinogram shape: (180, 4, 256)
          ...
          Reconstruction time: 0.xxx seconds

============================================================
  Test 5: GPU Performance Check
============================================================
  [✓ PASS] GPU acceleration performance

============================================================
  SUMMARY
============================================================
  All tests PASSED!
  MBIRJAX is correctly installed with CUDA support.
  The GPU backend is functional and performing reconstructions.

============================================================
```

## Troubleshooting

### No GPU Detected
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA toolkit version matches requirements (CUDA 12.x)
- Ensure the system has a compatible NVIDIA GPU

### Import Errors
- Reinstall the environment: `pixi install --manifest-path pixi_mbirjax.toml --force`
- Check for version conflicts in pixi.lock

### Reconstruction Fails
- Check GPU memory availability: `nvidia-smi`
- Try reducing sinogram size in tests
- Check for CUDA out-of-memory errors in logs

### Slow Performance
- Verify GPU is not thermally throttling
- Check if other processes are using the GPU
- Ensure the correct GPU is being used (multi-GPU systems)

## Environment Details

The `pixi_mbirjax.toml` environment includes:
- Python 3.10-3.11
- NumPy
- mbirjax with cuda12 extras (pulls in jax[cuda12])
- JupyterLab (for notebook testing)
- matplotlib (for visualization)

See `pixi_mbirjax.toml` for exact version specifications.
