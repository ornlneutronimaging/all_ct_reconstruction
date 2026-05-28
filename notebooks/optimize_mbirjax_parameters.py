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
    import matplotlib.pyplot as plt
    return glob, h5py, json, os, plt, re


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
    )
    return angles_deg, config, normalized_images_log


@app.cell
def _(
    angles_deg,
    bottom_line_slider,
    mo,
    normalized_images_log,
    plt,
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(first_image, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
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
    mo.vstack([top_line_slider, bottom_line_slider, z_range_slider])
    return bottom_line_slider, top_line_slider, z_range_slider


if __name__ == "__main__":
    app.run()
