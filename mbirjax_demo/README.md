# MBIRJAX CUDA Verification (Minimal Demo)

This folder contains a minimal CUDA verification workflow for mbirjax using:
- a command-line script for automated checks
- a Jupyter notebook for interactive inspection and visualization

## Purpose

The checks confirm that:
1. mbirjax, JAX, and NumPy import correctly
2. CUDA-capable GPUs are visible to JAX
3. JAX computations run on GPU
4. A synthetic CT reconstruction completes successfully
5. GPU performance is within a reasonable range

## Files

| File | Description |
|------|-------------|
| `test_mbirjax_cuda.py` | Command-line verification script |
| `test_mbirjax.ipynb` | Interactive notebook with plots and summaries |
| `README.md` | This file |
| `pixi.toml` | Environment definition and dependencies |

## Environment Setup

Install the environment defined in [pixi.toml](pixi.toml):

```bash
pixi install --manifest-path pixi.toml
```

## Run the Command-Line Script

```bash
pixi run --manifest-path pixi.toml python test_mbirjax_cuda.py
```

**Exit Codes:**
- `0` – all tests passed
- `1` – import errors or missing dependencies
- `2` – CUDA/GPU not available or not working
- `3` – reconstruction test failed
- `4` – GPU performance below expected

## Run the Notebook

Start JupyterLab from this folder and open [test_mbirjax.ipynb](test_mbirjax.ipynb):

```bash
pixi run --manifest-path pixi.toml jupyter lab --notebook-dir=.
```

The notebook walks through:
1. Package imports and version checks
2. GPU device detection
3. JAX GPU backend verification
4. Synthetic phantom + sinogram generation
5. MBIRJAX reconstruction + validation
6. Visualization of phantom, sinogram, and reconstruction
7. GPU performance test and summary

## What the Script Verifies

### 1) Package Imports
- `numpy`, `jax`, `mbirjax`

### 2) CUDA Device Detection
- enumerates JAX devices and reports GPU/CPU counts

### 3) JAX GPU Backend
- runs a matrix multiply on the GPU and checks device placement

### 4) MBIRJAX Reconstruction
- generates synthetic sinogram data
- reconstructs with `mj.ParallelBeamModel`
- checks shape and that output has no NaN/Inf

### 5) GPU Performance
- runs multiple reconstructions
- confirms timing is reasonable for the test size
- ensures computations actually execute on GPU

## Troubleshooting

### No GPU Detected
- Confirm NVIDIA drivers are installed (`nvidia-smi`)
- Verify CUDA 12-compatible GPU and driver stack

### Import Errors
- Reinstall: `pixi install --manifest-path pixi.toml --force`
- Check for dependency conflicts in pixi.lock

### Reconstruction Fails
- Ensure sufficient GPU memory
- Verify CUDA libraries are accessible in the environment

### Slow Performance
- Check for GPU contention or thermal throttling
- Confirm computations are placed on the GPU

## Environment Details

[pixi.toml](pixi.toml) defines the minimal CUDA-enabled stack, including:
- Python >= 3.12
- NumPy
- mbirjax with `cuda12` extras (pulls JAX CUDA runtime)
- JupyterLab + matplotlib for notebook usage
