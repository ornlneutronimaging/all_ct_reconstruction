"""
Image Processing Utilities for CT Reconstruction Pipeline

This module provides image processing functions for cleaning and preprocessing
images in the CT reconstruction workflow. Functions include outlier pixel
replacement and noise reduction techniques commonly needed for neutron imaging
data preparation.

Functions:
    replace_pixels: Replace outlier pixels using median filtering
    median_filter_3d: Parallel per-image median filter (multithreaded CPU / GPU)

Dependencies:
    - numpy: Numerical computations
    - scipy.ndimage: Image filtering operations

Author: CT Reconstruction Development Team
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tomopy
from scipy.ndimage import median_filter
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Accelerated median filter for a stack of images
# ---------------------------------------------------------------------------
# A per-image 2D median filter (size[0] == 1) is embarrassingly parallel along
# axis 0: each image is filtered independently. scipy's median_filter is
# single-threaded, so on a multi-core box the fastest path is simply to split
# the stack into contiguous chunks and run scipy on each chunk in parallel
# threads -- scipy releases the GIL during its C loop, so threads scale nearly
# linearly with no pickling and no extra dependency.
#
# On a machine with few CPU cores, a GPU (via JAX, an optional dependency) can
# win instead. We therefore default to the CPU-multithreaded path and only fall
# back to the GPU when cores are scarce (or when explicitly requested).
#
# All three paths are numerically identical to
# ``scipy.ndimage.median_filter(data, size=size)`` -- including scipy's default
# ``mode='reflect'`` boundary handling.

# Below this core count the GPU is preferred (when available) over CPU threads.
_CPU_CORE_THRESHOLD = 8

_JAX_GPU = None  # None = not probed yet, otherwise True/False


def _jax_gpu_available() -> bool:
    """Return True if JAX is importable and at least one GPU device is visible."""
    global _JAX_GPU
    if _JAX_GPU is None:
        try:
            import jax
            _JAX_GPU = any(d.platform == "gpu" for d in jax.devices())
        except Exception as e:  # JAX missing, no CUDA, driver mismatch, ...
            logging.info(f"GPU median filter unavailable ({e})")
            _JAX_GPU = False
    return _JAX_GPU


def _median_filter_cpu_mt(data: NDArray, size: tuple, workers: int) -> NDArray:
    """scipy median_filter over contiguous axis-0 chunks, one per thread."""
    idx = [c for c in np.array_split(np.arange(data.shape[0]), workers) if len(c)]
    with ThreadPoolExecutor(max_workers=len(idx)) as ex:
        parts = list(ex.map(
            lambda c: median_filter(data[c[0]:c[-1] + 1], size=size), idx))
    return np.concatenate(parts, axis=0)


def _median_filter_gpu(data: NDArray, size: tuple, chunk: int) -> NDArray:
    """JAX/GPU median filter over contiguous axis-0 chunks (bounds GPU memory)."""
    import jax
    import jax.numpy as jnp

    ky, kx = int(size[1]), int(size[2])
    py, px = ky // 2, kx // 2

    @jax.jit
    def _filt(x):
        xp = jnp.pad(x, ((0, 0), (py, py), (px, px)), mode="symmetric")
        h, w = x.shape[1], x.shape[2]
        nb = [xp[:, i:i + h, j:j + w] for i in range(ky) for j in range(kx)]
        return jnp.median(jnp.stack(nb, axis=0), axis=0)

    out = np.empty_like(data)
    for start in range(0, data.shape[0], chunk):
        end = min(start + chunk, data.shape[0])
        out[start:end] = np.asarray(_filt(jnp.asarray(data[start:end])))
    return out


def median_filter_3d(data: NDArray[np.floating],
                     size: tuple = (1, 3, 3),
                     backend: str = "auto",
                     workers: int = None,
                     chunk: int = 64) -> NDArray:
    """
    Apply a per-image 2D median filter to a stack of images, in parallel.

    Drop-in, numerically identical replacement for
    ``scipy.ndimage.median_filter(data, size=size)`` for the common CT case
    where ``size[0] == 1`` (each image along axis 0 filtered independently).

    Args:
        data: 3D array (n_images, height, width).
        size: Filter window; first element must be 1 for the parallel paths.
        backend: ``"cpu"`` (multithreaded scipy), ``"gpu"`` (JAX), ``"scipy"``
            (single-threaded reference), or ``"auto"`` (default): use CPU
            threads when enough cores are available, otherwise GPU if present,
            otherwise single-threaded scipy.
        workers: Thread count for the CPU path (defaults to ``os.cpu_count()``).
        chunk: Images processed per GPU batch (GPU path only).

    Returns:
        Filtered array, same shape and dtype as ``data``.
    """
    data = np.asarray(data)
    workers = workers or os.cpu_count() or 1

    # The parallel paths only apply to a stack filtered image-by-image.
    parallelizable = data.ndim == 3 and size[0] == 1

    # Resolve "auto" to a concrete backend.
    if backend == "auto":
        if not parallelizable or data.shape[0] < 2:
            backend = "scipy"
        elif workers >= _CPU_CORE_THRESHOLD:
            backend = "cpu"
        elif _jax_gpu_available():
            backend = "gpu"
        else:
            backend = "cpu"  # few cores, no GPU: threads still beat 1 thread

    try:
        if backend == "cpu" and parallelizable:
            return _median_filter_cpu_mt(data, size, workers)
        if backend == "gpu" and parallelizable and _jax_gpu_available():
            return _median_filter_gpu(data, size, chunk)
    except Exception as e:  # OOM, thread error, ... -> safe single-threaded path
        logging.warning(f"{backend} median filter failed ({e}); using scipy")

    return np.asarray(median_filter(data, size=size))


def replace_pixels(im: NDArray[np.floating], 
                   nbr_bins: int = 0, 
                   low_gate: int = 1, 
                   high_gate: int = 9, 
                   correct_radius: int = 1) -> NDArray[np.floating]:
    """
    Replace outlier pixels in an image using median filtering.
    
    Identifies pixels that fall outside specified histogram bins and replaces
    them with values from a median-filtered version of the image. This is
    commonly used to remove hot pixels, dead pixels, and other imaging
    artifacts in neutron CT data.
    
    Args:
        im: Input image array
        nbr_bins: Number of histogram bins for threshold calculation  
        low_gate: Lower histogram bin index for threshold
        high_gate: Upper histogram bin index for threshold
        correct_radius: Radius for median filter correction
        
    Returns:
        Image array with outlier pixels replaced
        
    Example:
        >>> import numpy as np
        >>> # Create test image with outliers
        >>> image = np.random.normal(100, 10, (256, 256))
        >>> image[50, 50] = 1000  # Hot pixel
        >>> image[100, 100] = 0   # Dead pixel
        >>> corrected = replace_pixels(image, nbr_bins=100)
        
    Note:
        The function uses histogram analysis to identify thresholds and
        replaces outlier pixels with median-filtered values to preserve
        local image structure while removing artifacts.
    """

    _, bin_edges = np.histogram(im.flatten(), bins=nbr_bins, density=False)
    thres_low = bin_edges[low_gate]
    thres_high = bin_edges[high_gate]

    y_coords, x_coords = np.nonzero(np.logical_or(im <= thres_low, 
                                                  im > thres_high))

    full_median_filter_corr_im = median_filter(im, size=correct_radius)
    for y, x in zip(y_coords, x_coords):
        im[y, x] = full_median_filter_corr_im[y, x]

    return im


def gamma_filter(
    arrays: NDArray[np.floating],
    threshold: int = -1,
    median_kernel: int = 5,
    axis: int = 0,
    max_workers: int = 0,
    selective_median_filter: bool = True,
    diff_tomopy: float = -1,
) -> NDArray[np.floating]:
    """Replace near-saturated pixels (from gamma radiation) with median values.

    Ported from imars3d.backend.corrections.gamma_filter. Uses
    tomopy.remove_outlier for the underlying median filtering.

    Args:
        arrays: 3D array of images (first dimension is rotation angle).
        threshold: Saturation threshold. -1 uses dtype max - 5.
        median_kernel: Size of the median filter kernel.
        axis: Axis along which to chunk for parallel filtering.
        max_workers: Number of cores (0 = all available minus 2).
        selective_median_filter: If True, only replace pixels above threshold.
        diff_tomopy: Outlier detection threshold for tomopy. Negative values
            use 20% of saturation intensity.

    Returns:
        Corrected 3D array of images.
    """
    if max_workers <= 0:
        max_workers = max(1, (os.cpu_count() or 1) - 2)

    try:
        saturation_intensity = np.iinfo(arrays.dtype).max
    except ValueError:
        saturation_intensity = 65535

    if threshold == -1:
        threshold = saturation_intensity - 5

    if diff_tomopy < 0:
        diff_tomopy = 0.2 * saturation_intensity

    arrays_filtered = tomopy.remove_outlier(
        arrays,
        dif=diff_tomopy,
        size=median_kernel,
        axis=axis,
        ncore=max_workers,
    )

    if selective_median_filter:
        arrays_filtered = np.where(
            arrays > threshold,
            arrays_filtered,
            arrays,
        )

    return arrays_filtered
