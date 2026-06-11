import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import glob
    import os
    import re
    import json
    import h5py
    import numpy as np
    import matplotlib
    import plotly.graph_objects as go
    from scipy.ndimage import rotate

    def mpl_colormap_to_plotly(name, n=64):
        cmap = matplotlib.colormaps[name]
        samples = cmap(np.linspace(0, 1, n))
        return [
            [i / (n - 1), f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"]
            for i, (r, g, b, _) in enumerate(samples)
        ]

    return glob, go, h5py, json, mpl_colormap_to_plotly, np, os, re, rotate


@app.cell
def _():
    import logging as _logging
    import getpass as _getpass
    import os as _os

    # per-user session log for this notebook; use the shared log folder when it
    # exists, otherwise fall back to ~/log
    _log_dir = "/SNS/VENUS/shared/log"
    if not _os.path.exists(_log_dir):
        _log_dir = _os.path.join(_os.path.expanduser("~"), "log")
        _os.makedirs(_log_dir, exist_ok=True)

    _user_id = _getpass.getuser()
    log_file_path = _os.path.join(
        _log_dir, f"remove_strips_using_bm3dornl_{_user_id}.log"
    )

    # a dedicated, non-propagating logger so it is independent of the root logger
    # (which the reconstruction-evaluation module reconfigures on import and would
    # otherwise redirect or duplicate these messages)
    logger = _logging.getLogger("remove_strips_using_bm3dornl")
    logger.setLevel(_logging.INFO)
    logger.propagate = False
    # drop any handler left from a previous run of this cell to avoid duplicate
    # log lines
    for _handler in list(logger.handlers):
        logger.removeHandler(_handler)
        _handler.close()
    _file_handler = _logging.FileHandler(log_file_path, mode="w")
    _file_handler.setFormatter(
        _logging.Formatter("[%(levelname)s] - %(asctime)s - %(message)s")
    )
    logger.addHandler(_file_handler)
    logger.info("*** remove_strips_using_bm3dornl session started ***")
    return log_file_path, logger


@app.cell
def _(glob, os):
    def list_accessible_ipts(base_path="/SNS/VENUS"):
        """Return {ipts_name: full_path} for IPTS-* folders the user can read."""
        accessible = {}
        for path in glob.glob(os.path.join(base_path, "IPTS-*")):
            if os.path.isdir(path) and os.access(path, os.R_OK):
                accessible[os.path.basename(path)] = path

        def ipts_number(name):
            try:
                return int(name.split("-", 1)[1])
            except (IndexError, ValueError):
                return -1

        return dict(
            sorted(accessible.items(), key=lambda kv: ipts_number(kv[0]), reverse=True)
        )

    accessible_ipts = list_accessible_ipts()
    return (accessible_ipts,)


@app.cell
def _(accessible_ipts, mo):
    mo.stop(
        len(accessible_ipts) == 0,
        mo.md("**No accessible IPTS folders found in `/SNS/VENUS`.**"),
    )

    _default_ipts = "IPTS-36573"  # debugging default
    ipts_selector = mo.ui.dropdown(
        options=accessible_ipts,
        value=_default_ipts if _default_ipts in accessible_ipts else None,
        label="**Select an IPTS:**",
        searchable=True,
    )
    ipts_selector
    return (ipts_selector,)


@app.cell
def _(glob, ipts_selector, mo, os, re):
    mo.stop(
        ipts_selector.value is None,
        mo.md("*Select an IPTS above to list its HDF5 files.*"),
    )

    shared_path = os.path.join(ipts_selector.value, "shared")
    start_path = shared_path if os.path.isdir(shared_path) else ipts_selector.value

    step_pattern = re.compile(r"_step(\d+)\.hdf5$")

    def step_number(path):
        match = step_pattern.search(os.path.basename(path))
        return int(match.group(1)) if match else None

    step_hdf5_files = sorted(
        path
        for path in glob.glob(
            os.path.join(start_path, "**", "*_step*.hdf5"), recursive=True
        )
        if (number := step_number(path)) is not None and number >= 2
    )

    mo.stop(
        len(step_hdf5_files) == 0,
        mo.md(f"*No `_step#.hdf5` files (step ≥ 2) found under `{start_path}`.*"),
    )

    hdf5_selector = mo.ui.dropdown(
        options={os.path.relpath(p, start_path): p for p in step_hdf5_files},
        label="**Select an HDF5 file (`_step#.hdf5`, step ≥ 2):**",
        searchable=True,
    )
    hdf5_selector
    return (hdf5_selector,)


@app.cell
def _(hdf5_selector, mo):
    mo.stop(
        hdf5_selector.value is None,
        mo.md("*Select an HDF5 file above to continue.*"),
    )

    selected_hdf5_file = hdf5_selector.value
    mo.md(f"Selected HDF5 file: `{selected_hdf5_file}`")
    return (selected_hdf5_file,)


@app.cell
def _(h5py, json, mo, selected_hdf5_file):
    with h5py.File(selected_hdf5_file, "r") as f:
        config_raw = f["metadata/config"][()] if "metadata/config" in f else None
        config = json.loads(config_raw) if config_raw is not None else None
        normalized_images_log = (
            f["raw/normalized_images_log"][:]
            if "raw/normalized_images_log" in f
            else None
        )
        angles_deg = f["angles/deg"][:] if "angles/deg" in f else None

    mo.stop(
        config is None and normalized_images_log is None,
        mo.md(
            "**Selected file has neither `metadata/config` nor "
            "`raw/normalized_images_log`.**"
        ),
    )

    def _format_config(cfg):
        if cfg is None:
            return "missing"
        lines = []
        for k, v in cfg.items():
            lines.append(f"  - **{k}**: `{v}`")
        return "\n".join(lines)

    _info_content = mo.vstack(
        [
            mo.md(
                f"""
                Loaded from `{selected_hdf5_file}`:

                - **config** (`metadata/config`): {"loaded" if config is not None else "missing"}
                - **normalized_images_log** (`raw/normalized_images_log`): {
                    normalized_images_log.shape
                    if normalized_images_log is not None
                    else "missing"
                }
                - **angles** (`angles/deg`): {
                    len(angles_deg) if angles_deg is not None else "missing"
                }

                {("**Config dictionary:**\n\n" + _format_config(config)) if config is not None else ""}
                """
            ),
        ]
    ).style(
        {
            "background-color": "#eef2f7",
            "color": "#1a1a1a",
            "padding": "1rem",
            "border-radius": "8px",
            "border": "1px solid #c5d0dd",
        }
    )

    mo.accordion({"ℹ️ Infos": _info_content})
    return angles_deg, config, normalized_images_log


@app.cell
def _(mo, normalized_images_log):
    # the 3D stack is (n_angles, n_slices, n_det_channels); a sinogram for a
    # given slice (detector row) is the (n_angles, n_det_channels) plane, which
    # is what bm3dornl operates on to remove streaks/rings
    n_angles, n_slices, n_det_channels = normalized_images_log.shape

    slice_slider = mo.ui.slider(
        start=0,
        stop=n_slices - 1,
        value=0,
        label="Slice (detector row):",
        show_value=True,
        full_width=True,
    )

    # z-range bounds come from the whole volume so moving the slice slider does
    # not rescale the colour mapping under the user
    intensity_min = float(normalized_images_log.min())
    intensity_max = float(normalized_images_log.max())
    _z_step = (
        (intensity_max - intensity_min) / 200
        if intensity_max > intensity_min
        else 1.0
    )
    z_range_slider = mo.ui.range_slider(
        start=intensity_min,
        stop=intensity_max,
        step=_z_step,
        value=[intensity_min, intensity_max],
        label="z range (intensity):",
        show_value=True,
        orientation="vertical",
    )
    colormap_selector = mo.ui.dropdown(
        options=[
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "jet",
            "turbo",
            "seismic",
        ],
        value="gray",
        label="Colormap:",
        searchable=True,
    )
    return colormap_selector, slice_slider, z_range_slider


@app.cell
def _(
    angles_deg,
    colormap_selector,
    go,
    mo,
    mpl_colormap_to_plotly,
    normalized_images_log,
    np,
    slice_slider,
    z_range_slider,
):
    mo.stop(
        normalized_images_log is None,
        mo.md("**No `raw/normalized_images_log` data to display.**"),
    )

    slice_index = slice_slider.value
    sinogram = normalized_images_log[:, slice_index, :]
    n_angles, n_cols = sinogram.shape
    vmin, vmax = z_range_slider.value

    # subsample the displayed heatmap when the sinogram is wide so the preview
    # stays responsive; pass original-coordinate x/y arrays so the axes keep
    # showing the full sinogram size
    downsample = 10 if n_cols > 1000 else 1
    if downsample > 1:
        _heatmap = dict(
            z=sinogram[:, ::downsample],
            x=np.arange(0, n_cols, downsample),
            y=np.arange(n_angles),
        )
    else:
        _heatmap = dict(z=sinogram, x=np.arange(n_cols), y=np.arange(n_angles))

    _angle_range = (
        f" ({float(angles_deg[0]):.1f}° → {float(angles_deg[-1]):.1f}°)"
        if angles_deg is not None and len(angles_deg)
        else ""
    )

    _fig = go.Figure(
        go.Heatmap(
            **_heatmap,
            colorscale=mpl_colormap_to_plotly(colormap_selector.value),
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="intensity"),
        )
    )
    _fig.update_layout(
        title=f"Sinogram of slice {slice_index}{_angle_range}",
        height=600,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    _fig.update_xaxes(title_text="detector channel (column)", range=[-0.5, n_cols - 0.5])
    _fig.update_yaxes(title_text="projection (angle index)", range=[n_angles - 0.5, -0.5])

    mo.vstack(
        [
            slice_slider,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [z_range_slider.style({"height": "500px"}), colormap_selector],
                        align="center",
                    ),
                ],
                align="center",
                justify="start",
                gap=1,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
