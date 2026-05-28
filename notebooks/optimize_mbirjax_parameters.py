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
    import matplotlib.pyplot as plt
    from scipy.ndimage import rotate
    return glob, h5py, json, np, os, plt, re, rotate


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

    _default_ipts = "IPTS-35712"  # debugging default
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

    mo.vstack(
        [
            mo.md("### **ℹ️ Infos**"),
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
    return angles_deg, config, normalized_images_log


@app.cell
def _(
    angles_deg,
    bottom_line_slider,
    colormap_selector,
    mbirjax_widgets,
    mo,
    normalized_images_log,
    np,
    perform_tilt_switch,
    plt,
    rotate,
    show_grid_toggle,
    tilt_slider,
    top_line_slider,
    z_range_slider,
):
    mo.stop(
        normalized_images_log is None,
        mo.md("**No `raw/normalized_images_log` data to display.**"),
    )

    band_half = 5

    first_image = normalized_images_log[0]
    first_angle = (
        float(angles_deg[0])
        if angles_deg is not None and len(angles_deg)
        else None
    )
    vmin, vmax = z_range_slider.value

    tilt_angle = tilt_slider.value
    perform_tilt = perform_tilt_switch.value

    # rotate(+tilt) straightens a feature aligned with the grid drawn at -tilt
    display_image = (
        rotate(
            first_image,
            angle=tilt_angle,
            reshape=False,
            order=1,
            mode="nearest",
        )
        if perform_tilt and tilt_angle != 0
        else first_image
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        display_image,
        cmap=colormap_selector.value,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.axhspan(
        top_line_slider.value - band_half,
        top_line_slider.value + band_half,
        color="red",
        alpha=0.25,
    )
    ax.axhline(top_line_slider.value, color="red", linewidth=1.0)
    ax.axhspan(
        bottom_line_slider.value - band_half,
        bottom_line_slider.value + band_half,
        color="cyan",
        alpha=0.25,
    )
    ax.axhline(bottom_line_slider.value, color="cyan", linewidth=1.0)

    center_column = first_image.shape[1] / 2
    det_channel_offset = mbirjax_widgets.value.get("det_channel_offset")
    if det_channel_offset is not None:
        ax.axvline(
            center_column + det_channel_offset,
            color="white",
            linestyle="--",
            linewidth=1.0,
        )

    if show_grid_toggle.value:
        # grid is tilted by tilt_angle until the tilt is performed, then straight
        grid_angle = 0.0 if perform_tilt else tilt_angle
        n_rows, n_cols = first_image.shape
        center = np.array([n_cols / 2.0, n_rows / 2.0])
        theta = np.deg2rad(-grid_angle)
        # display: +x to the right, +y downward; +angle is up so row decreases
        along = np.array([np.cos(theta), -np.sin(theta)])
        across = np.array([np.sin(theta), np.cos(theta)])
        diag = float(np.hypot(n_rows, n_cols))
        spacing = max(n_rows, n_cols) / 10.0
        n_lines = int(diag / spacing) + 1
        for k in range(-n_lines, n_lines + 1):
            for direction, offset_dir in ((along, across), (across, along)):
                base = center + k * spacing * offset_dir
                p0 = base - diag * direction
                p1 = base + diag * direction
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color="yellow",
                    linewidth=0.5,
                    alpha=0.6,
                )
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)

    ax.set_title(
        f"Projection at {first_angle:.3f}°"
        if first_angle is not None
        else "First projection"
    )
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig
    return (first_image,)


@app.cell
def _(mo, normalized_images_log):
    image_height = normalized_images_log.shape[1]

    _first_image = normalized_images_log[0]
    intensity_min = float(_first_image.min())
    intensity_max = float(_first_image.max())
    _z_step = (
        (intensity_max - intensity_min) / 200
        if intensity_max > intensity_min
        else 1.0
    )

    top_line_slider = mo.ui.slider(
        start=0,
        stop=image_height - 1,
        value=int(image_height * 0.1),
        label="Top line (row):",
        show_value=True,
        full_width=True,
    )
    bottom_line_slider = mo.ui.slider(
        start=0,
        stop=image_height - 1,
        value=int(image_height * 0.9),
        label="Bottom line (row):",
        show_value=True,
        full_width=True,
    )
    z_range_slider = mo.ui.range_slider(
        start=intensity_min,
        stop=intensity_max,
        step=_z_step,
        value=[intensity_min, intensity_max],
        label="z range (intensity):",
        show_value=True,
        full_width=True,
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
    tilt_slider = mo.ui.slider(
        start=-5.0,
        stop=5.0,
        step=0.01,
        value=0.0,
        label="Tilt (°):",
        show_value=True,
        full_width=True,
    )
    show_grid_toggle = mo.ui.switch(label="Show grid")
    perform_tilt_switch = mo.ui.switch(label="Perform tilt")
    mo.vstack(
        [
            top_line_slider,
            bottom_line_slider,
            z_range_slider,
            colormap_selector,
            tilt_slider,
            mo.hstack(
                [show_grid_toggle, perform_tilt_switch],
                justify="start",
                gap=2,
            ),
        ]
    )
    return (
        bottom_line_slider,
        colormap_selector,
        perform_tilt_switch,
        show_grid_toggle,
        tilt_slider,
        top_line_slider,
        z_range_slider,
    )


@app.cell
def _(config, mo):
    mbirjax_params = (config or {}).get("mbirjax_config", {})

    mo.stop(
        not mbirjax_params,
        mo.md("*No `mbirjax_config` parameters found in the config.*"),
    )

    def make_mbirjax_widget(name, value):
        if isinstance(value, bool):
            return mo.ui.checkbox(value=value, label=name)
        if isinstance(value, int):
            return mo.ui.number(value=value, step=1, label=name)
        if isinstance(value, float):
            return mo.ui.number(value=value, step=0.1, label=name)
        return mo.ui.text(value=str(value), label=name)

    mbirjax_widgets = mo.ui.dictionary(
        {
            name: make_mbirjax_widget(name, value)
            for name, value in mbirjax_params.items()
        }
    )
    mo.vstack(
        [
            mo.md("### **⚙️ MBIRJAX parameters**"),
            *mbirjax_widgets.elements.values(),
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
    return (mbirjax_widgets,)


if __name__ == "__main__":
    app.run()
