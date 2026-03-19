"""Center of rotation (COR) and tilt correction utilities.

Ported from neutompy.preproc.preproc (find_COR, correction_COR).
Uses skimage.transform.rotate instead of SimpleITK for image rotation.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from numpy.typing import NDArray
from skimage.transform import rotate
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_COR(
    proj_0: NDArray,
    proj_180: NDArray,
    ystep: int = 5,
    show_results: bool = True,
    rois: list[tuple[int, int]] | None = None,
) -> tuple[float, float]:
    """Estimate rotation axis offset and tilt from 0/180-degree projections.

    The offset is found by minimizing the RMSE between proj_0 and the
    horizontally-flipped proj_180 across user-specified ROI rows.

    Args:
        proj_0: Projection at 0 degrees.
        proj_180: Projection at 180 degrees.
        ystep: Row step size inside each ROI.
        show_results: If True, display diagnostic matplotlib figures.
        rois: List of (ymin, ymax) tuples defining row ranges. Required.

    Returns:
        (middle_shift, theta) – horizontal pixel shift and tilt angle in degrees.

    Raises:
        ValueError: If rois is not provided.
    """
    if rois is None:
        raise ValueError("rois must be provided")

    proj_0 = proj_0.astype(np.float32)
    proj_180 = proj_180.astype(np.float32)

    nd = proj_0.shape[1]  # columns
    nz = proj_0.shape[0]  # rows

    slices = np.array([], dtype=np.int32)

    tmin = -nd // 2
    tmax = nd - nd // 2

    for ymin, ymax in rois:
        aus = np.arange(ymin, ymax + 1, ystep)
        slices = np.concatenate((slices, aus), axis=0)

    shift = np.zeros(slices.size)
    proj_flip = proj_180[:, ::-1]

    for z, slc in enumerate(slices):
        minimum = 1e7
        index_min = 0
        posz = np.round(slc).astype(np.int32)

        for t in range(tmin, tmax + 1):
            rmse = np.square(np.roll(proj_0[posz], t, axis=0) - proj_flip[posz]).sum() / nd
            if rmse <= minimum:
                minimum = rmse
                index_min = t

        shift[z] = index_min

    # linear fit
    par = np.polyfit(slices, shift, deg=1)
    m, q = par[0], par[1]

    theta = np.rad2deg(np.arctan(0.5 * m))
    middle_shift = int(np.round(m * nz * 0.5 + q)) // 2
    offset = int(np.round(m * nz * 0.5 + q)) * 0.5

    logger.info(f"COR found: offset={offset}, tilt angle={theta:.4f}")

    if show_results:
        _plot_cor_results(proj_0, proj_180, proj_flip, shift, slices, m, q, nd, nz, offset, theta, middle_shift)

    return middle_shift, theta


def correction_COR(
    norm_proj: NDArray,
    proj_0: NDArray,
    proj_180: NDArray,
    show_opt: str = "mean",
    shift: int | None = None,
    theta: float | None = None,
    ystep: int = 5,
    show_results: bool = False,
    rois: list[tuple[int, int]] | None = None,
) -> NDArray:
    """Correct rotation axis misalignment for a projection stack.

    If *shift* and *theta* are both None they are estimated from the 0/180
    degree pair via :func:`find_COR`.

    Args:
        norm_proj: 3-D projection stack (theta, y, x).
        proj_0: Projection at 0 degrees.
        proj_180: Projection at 180 degrees.
        show_opt: Image used for ROI display ('mean', 'std', 'zero', 'pi').
        shift: Known horizontal shift (pixels). None → estimate.
        theta: Known tilt angle (degrees). None → estimate.
        ystep: Row step for find_COR.
        show_results: If True, display diagnostic matplotlib figures.
        rois: Explicit ROI list for find_COR.

    Returns:
        Corrected projection stack (modified in place and returned).

    Raises:
        ValueError: If only one of shift/theta is provided.
    """
    if (shift is None) != (theta is None):
        raise ValueError("shift and theta must both be provided, or both be None")

    if shift is None and theta is None:
        if show_opt == "mean":
            proj2show = norm_proj.mean(axis=0, dtype=np.float32)
        elif show_opt == "std":
            proj2show = norm_proj.std(axis=0, dtype=np.float32)
        elif show_opt == "zero":
            proj2show = proj_0
        elif show_opt == "pi":
            proj2show = proj_180
        else:
            raise ValueError("show_opt must be 'mean', 'std', 'zero', or 'pi'.")

        shift, theta = find_COR(
            proj_0, proj_180, ystep=ystep, show_results=show_results, rois=rois
        )

    logger.info(f"Correcting rotation axis: shift={shift}, theta={theta:.4f}")
    for s in tqdm(range(norm_proj.shape[0]), unit=" images"):
        norm_proj[s, :, :] = np.roll(
            rotate(norm_proj[s, :, :], theta, preserve_range=True, order=1, mode="edge"),
            shift,
            axis=1,
        )

    return norm_proj


def _plot_cor_results(proj_0, proj_180, proj_flip, shift, slices, m, q, nd, nz, offset, theta, middle_shift):
    """Plot diagnostic figures for COR detection."""
    p0_r = np.roll(
        rotate(proj_0, theta, preserve_range=True, order=1, mode="edge"),
        middle_shift, axis=1,
    )
    p90_r = np.roll(
        rotate(proj_180, theta, preserve_range=True, order=1, mode="edge"),
        middle_shift, axis=1,
    )

    # Figure 1: before correction
    plt.figure("Analysis of the rotation axis position", figsize=(14, 5), dpi=96)
    plt.subplots_adjust(wspace=0.5)
    ax1 = plt.subplot(1, 2, 1)

    diff = proj_0 - proj_flip
    mu, s = np.median(diff), diff.std()
    plt.imshow(diff, cmap="gray", vmin=mu - s, vmax=mu + s)
    info_cor = f"offset = {offset:.2f}\n       \u03b8 = {theta:.3f}"
    ax1.add_artist(AnchoredText(info_cor, loc=2))
    plt.title(r"$P_0 - P^{flipped}_{\pi}$ before correction")
    plt.colorbar(fraction=0.046, pad=0.04)

    zaxis = np.arange(0, nz)
    plt.plot(nd * 0.5 - 0.5 * m * zaxis - 0.5 * q, zaxis, "b-")
    plt.plot(nd * 0.5 - 0.5 * shift, slices, "r.", markersize=3)
    plt.plot([0.5 * nd, 0.5 * nd], [0, nz - 1], "k--")

    ax2 = plt.subplot(1, 2, 2)
    info_fit = f"shift = {m:.3f}*y + {q:.3f}"
    ax2.add_artist(AnchoredText(info_fit, loc=9))
    plt.plot(zaxis, m * zaxis + q, "b-", label="fit")
    plt.plot(slices, shift, "r.", label="data")
    plt.xlabel("$y$")
    plt.ylabel("shift")
    plt.title("Fit result")
    plt.legend()

    # Figure 2: after correction
    plt.figure("Results of the rotation axis correction", figsize=(14, 5), dpi=96)
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(wspace=0.5)

    diff2 = np.nan_to_num(p0_r - p90_r[:, ::-1])
    mu, s = np.nanmedian(diff2), diff2.std()
    plt.imshow(diff2, cmap="gray", vmin=mu - s, vmax=mu + s)
    plt.title(r"$P_0 - P^{flipped}_{\pi}$ after correction")
    plt.colorbar(fraction=0.046, pad=0.04)

    ax3 = plt.subplot(1, 2, 2)
    nbins = 1000
    row_marg = int(0.1 * diff2.shape[0])
    col_marg = int(0.1 * diff2.shape[1])
    absdif = np.abs(diff2)[row_marg:-row_marg, col_marg:-col_marg]
    binning, width = np.linspace(absdif.min(), absdif.max(), nbins, retstep=True)
    cc, edge = np.histogram(absdif, bins=binning)
    plt.bar(edge[:-1] + width * 0.5, cc, width, color="C3", edgecolor="k", log=True)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.xlim([0.01, np.nanmax(absdif)])
    plt.xlabel("Residuals")
    plt.ylabel("Entries")
    plt.title("Histogram of residuals")

    res = np.abs(diff2).mean()
    info_res = r"$||P_0 - P^{flipped}_{\pi}||_1 / N_{pixel}$ = " + f"{res:.4f}"
    ax3.add_artist(AnchoredText(info_res, loc=1))

    plt.show(block=False)
