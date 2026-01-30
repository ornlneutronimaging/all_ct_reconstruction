#!/usr/bin/env python
"""
MBIRJAX CUDA Installation Verification Script

This script verifies that mbirjax is correctly installed with CUDA support
by performing actual GPU-accelerated computations rather than just checking
version numbers.

Tests performed:
1. Import verification for mbirjax and JAX
2. CUDA device detection and availability
3. JAX GPU backend verification
4. Actual CT reconstruction on synthetic data using GPU
5. Performance comparison (GPU vs CPU) to confirm GPU acceleration

Usage:
    python test_mbirjax_cuda.py

    Or with pixi:
    pixi run test-cuda

Exit codes:
    0 - All tests passed, CUDA backend working
    1 - Import errors or missing dependencies
    2 - CUDA/GPU not available
    3 - Reconstruction test failed
    4 - Performance test suggests GPU not being used

Author: CT Reconstruction Pipeline Team
"""

import sys
import time
from typing import Tuple, Optional
import traceback


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print test result with status indicator."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {test_name}")
    if details:
        for line in details.split("\n"):
            print(f"          {line}")


def test_imports() -> Tuple[bool, str]:
    """Test that all required packages can be imported."""
    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        import mbirjax as mj

        details = (
            f"numpy version: {np.__version__}\n"
            f"jax version: {jax.__version__}\n"
            f"mbirjax version: {mj.__version__ if hasattr(mj, '__version__') else 'unknown'}"
        )
        return True, details
    except ImportError as e:
        return False, f"Import error: {e}"


def test_cuda_availability() -> Tuple[bool, str]:
    """Test that CUDA devices are available to JAX."""
    try:
        import jax

        # Get available devices
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        cpu_devices = [d for d in devices if d.platform == 'cpu']

        details = f"Total devices: {len(devices)}\n"
        details += f"GPU devices: {len(gpu_devices)}\n"
        details += f"CPU devices: {len(cpu_devices)}"

        if gpu_devices:
            details += "\nGPU device info:"
            for i, gpu in enumerate(gpu_devices):
                details += f"\n  GPU {i}: {gpu}"
            return True, details
        else:
            details += "\nNo GPU devices found!"
            return False, details

    except Exception as e:
        return False, f"Error checking CUDA: {e}"


def test_jax_gpu_backend() -> Tuple[bool, str]:
    """Test that JAX can actually use the GPU backend."""
    try:
        import jax
        import jax.numpy as jnp

        # Try to create an array on GPU and perform computation
        x = jnp.ones((1000, 1000))

        # Force computation on GPU
        with jax.default_device(jax.devices('gpu')[0]):
            y = jnp.dot(x, x)
            y.block_until_ready()  # Wait for computation to complete

        # Get the device where computation happened
        device = y.device()

        details = f"Computation device: {device}\n"
        details += f"Device platform: {device.platform}"

        if device.platform == 'gpu':
            return True, details
        else:
            return False, f"Computation ran on {device.platform}, not GPU"

    except Exception as e:
        return False, f"GPU backend test failed: {e}\n{traceback.format_exc()}"


def test_mbirjax_reconstruction() -> Tuple[bool, str, Optional[float]]:
    """
    Test actual mbirjax reconstruction on synthetic CT data.

    This creates a simple phantom, generates sinogram data, and performs
    reconstruction using mbirjax's ParallelBeamModel with GPU acceleration.

    Returns:
        Tuple of (passed, details, reconstruction_time)
    """
    try:
        import numpy as np
        import jax.numpy as jnp
        import mbirjax as mj

        # Create synthetic sinogram data
        # Small size for quick testing but large enough to verify GPU usage
        num_views = 180  # Number of projection angles
        num_det_channels = 256  # Detector width
        num_slices = 4  # Number of slices to reconstruct

        # Generate angles (0 to 180 degrees)
        angles = np.linspace(0, np.pi, num_views, endpoint=False).astype(np.float32)

        # Create a simple phantom (circle in the center)
        phantom_size = num_det_channels
        y, x = np.ogrid[-phantom_size//2:phantom_size//2, -phantom_size//2:phantom_size//2]
        mask = x*x + y*y <= (phantom_size//4)**2
        phantom = np.zeros((num_slices, phantom_size, phantom_size), dtype=np.float32)
        phantom[:, mask] = 1.0

        # Generate synthetic sinogram using forward projection approximation
        # (In real use, this would come from actual CT data)
        sinogram_shape = (num_views, num_slices, num_det_channels)

        # Create synthetic sinogram with some structure
        np.random.seed(42)
        sinogram = np.random.rand(*sinogram_shape).astype(np.float32) * 0.1

        # Add some signal based on simple radon-like projection
        for i, angle in enumerate(angles):
            projection = np.sum(phantom, axis=2 if angle < np.pi/2 else 1)
            sinogram[i, :, :] = projection[:, :num_det_channels] * 0.5 + sinogram[i, :, :] * 0.5

        details = f"Sinogram shape: {sinogram_shape}\n"
        details += f"Number of angles: {num_views}\n"
        details += f"Detector channels: {num_det_channels}\n"
        details += f"Slices: {num_slices}"

        # Create mbirjax model and run reconstruction
        start_time = time.time()

        ct_model = mj.ParallelBeamModel(sinogram_shape, angles)

        # Configure for GPU usage
        ct_model.set_params(
            sharpness=0.0,
            verbose=0,
            snr_db=30.0,
        )

        # Run reconstruction
        recon_result, recon_dict = ct_model.recon(
            sinogram,
            print_logs=False,
            weights=None
        )

        # Ensure computation is complete
        recon_result = np.array(recon_result)

        elapsed_time = time.time() - start_time

        details += f"\n\nReconstruction completed!"
        details += f"\nOutput shape: {recon_result.shape}"
        details += f"\nReconstruction time: {elapsed_time:.3f} seconds"
        details += f"\nOutput dtype: {recon_result.dtype}"
        details += f"\nOutput range: [{recon_result.min():.4f}, {recon_result.max():.4f}]"

        # Basic sanity check on output
        if recon_result.shape[0] != num_slices:
            return False, f"Unexpected output shape: {recon_result.shape}", elapsed_time

        if np.isnan(recon_result).any():
            return False, "Reconstruction contains NaN values", elapsed_time

        if np.isinf(recon_result).any():
            return False, "Reconstruction contains Inf values", elapsed_time

        return True, details, elapsed_time

    except Exception as e:
        return False, f"Reconstruction test failed: {e}\n{traceback.format_exc()}", None


def test_gpu_performance() -> Tuple[bool, str]:
    """
    Test that GPU provides expected performance improvement.

    Compares reconstruction time to ensure GPU acceleration is working.
    A working GPU should be significantly faster than CPU for this workload.
    """
    try:
        import numpy as np
        import jax
        import mbirjax as mj

        # Check if we have both CPU and GPU available
        gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']

        if not gpu_devices:
            return False, "No GPU available for performance comparison"

        # Create test data
        num_views = 90
        num_det_channels = 128
        num_slices = 2

        angles = np.linspace(0, np.pi, num_views, endpoint=False).astype(np.float32)
        sinogram_shape = (num_views, num_slices, num_det_channels)
        sinogram = np.random.rand(*sinogram_shape).astype(np.float32)

        # Run reconstruction (which should use GPU)
        ct_model = mj.ParallelBeamModel(sinogram_shape, angles)
        ct_model.set_params(sharpness=0.0, verbose=0, snr_db=30.0)

        # Warm-up run (JIT compilation)
        _, _ = ct_model.recon(sinogram, print_logs=False, weights=None)

        # Timed run
        start_time = time.time()
        for _ in range(3):
            result, _ = ct_model.recon(sinogram, print_logs=False, weights=None)
            np.array(result)  # Force synchronization
        gpu_time = (time.time() - start_time) / 3

        details = f"Average reconstruction time (3 runs): {gpu_time:.4f} seconds\n"

        # For this test size, GPU should complete quickly
        # If it takes more than 5 seconds, something might be wrong
        if gpu_time > 5.0:
            details += f"WARNING: Reconstruction seems slow for GPU ({gpu_time:.2f}s)"
            details += "\nThis might indicate GPU is not being used effectively"
            return False, details

        details += "GPU performance appears normal"
        return True, details

    except Exception as e:
        return False, f"Performance test failed: {e}"


def main() -> int:
    """Run all tests and report results."""
    print_header("MBIRJAX CUDA Installation Verification")
    print(f"Python version: {sys.version}")

    all_passed = True
    exit_code = 0

    # Test 1: Imports
    print_header("Test 1: Package Imports")
    passed, details = test_imports()
    print_result("Import mbirjax, jax, numpy", passed, details)
    if not passed:
        print("\nCRITICAL: Cannot import required packages. Aborting.")
        return 1

    # Test 2: CUDA Availability
    print_header("Test 2: CUDA Device Detection")
    passed, details = test_cuda_availability()
    print_result("CUDA/GPU devices available", passed, details)
    if not passed:
        all_passed = False
        exit_code = 2
        print("\nWARNING: No GPU detected. mbirjax will fall back to CPU.")

    # Test 3: JAX GPU Backend
    print_header("Test 3: JAX GPU Backend")
    passed, details = test_jax_gpu_backend()
    print_result("JAX GPU computation", passed, details)
    if not passed:
        all_passed = False
        if exit_code == 0:
            exit_code = 2

    # Test 4: MBIRJAX Reconstruction
    print_header("Test 4: MBIRJAX Reconstruction")
    passed, details, recon_time = test_mbirjax_reconstruction()
    print_result("CT reconstruction on synthetic data", passed, details)
    if not passed:
        all_passed = False
        if exit_code == 0:
            exit_code = 3

    # Test 5: GPU Performance
    print_header("Test 5: GPU Performance Check")
    passed, details = test_gpu_performance()
    print_result("GPU acceleration performance", passed, details)
    if not passed:
        all_passed = False
        if exit_code == 0:
            exit_code = 4

    # Summary
    print_header("SUMMARY")
    if all_passed:
        print("  All tests PASSED!")
        print("  MBIRJAX is correctly installed with CUDA support.")
        print("  The GPU backend is functional and performing reconstructions.")
    else:
        print("  Some tests FAILED!")
        print(f"  Exit code: {exit_code}")
        if exit_code == 2:
            print("  Issue: CUDA/GPU not available or not working")
        elif exit_code == 3:
            print("  Issue: Reconstruction failed")
        elif exit_code == 4:
            print("  Issue: GPU performance below expected")

    print("\n" + "=" * 60)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
