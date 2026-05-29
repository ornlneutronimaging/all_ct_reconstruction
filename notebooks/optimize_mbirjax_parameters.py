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
    go,
    mbirjax_widgets,
    mo,
    mpl_colormap_to_plotly,
    normalized_images_log,
    np,
    perform_tilt_switch,
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

    from __code.config import MARIMO_TEST_RECONSTRUCTION_WIDTH
    band_half = int(MARIMO_TEST_RECONSTRUCTION_WIDTH / 2)

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

    n_rows, n_cols = first_image.shape

    fig = go.Figure(
        go.Heatmap(
            z=display_image,
            colorscale=mpl_colormap_to_plotly(colormap_selector.value),
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="intensity"),
        )
    )

    fig.add_hrect(
        y0=top_line_slider.value - band_half,
        y1=top_line_slider.value + band_half,
        fillcolor="red",
        opacity=0.25,
        line_width=0,
    )
    fig.add_hline(y=top_line_slider.value, line_color="red", line_width=1.0)
    fig.add_hrect(
        y0=bottom_line_slider.value - band_half,
        y1=bottom_line_slider.value + band_half,
        fillcolor="cyan",
        opacity=0.25,
        line_width=0,
    )
    fig.add_hline(y=bottom_line_slider.value, line_color="cyan", line_width=1.0)

    center_column = first_image.shape[1] / 2
    det_channel_offset = mbirjax_widgets.value.get("det_channel_offset (rotation center column offset)", 0)
    if det_channel_offset is not None:
        fig.add_vline(
            x=center_column + det_channel_offset,
            line_color="white",
            line_dash="dash",
            line_width=1.0,
        )

    if show_grid_toggle.value:
        # grid is tilted by tilt_angle until the tilt is performed, then straight
        grid_angle = 0.0 if perform_tilt else tilt_angle
        center = np.array([n_cols / 2.0, n_rows / 2.0])
        theta = np.deg2rad(-grid_angle)
        # display: +x to the right, +y downward; +angle is up so row decreases
        along = np.array([np.cos(theta), -np.sin(theta)])
        across = np.array([np.sin(theta), np.cos(theta)])
        diag = float(np.hypot(n_rows, n_cols))
        spacing = max(n_rows, n_cols) / 10.0
        n_lines = int(diag / spacing) + 1
        grid_x = []
        grid_y = []
        for k in range(-n_lines, n_lines + 1):
            for direction, offset_dir in ((along, across), (across, along)):
                base = center + k * spacing * offset_dir
                p0 = base - diag * direction
                p1 = base + diag * direction
                grid_x += [p0[0], p1[0], None]
                grid_y += [p0[1], p1[1], None]
        fig.add_trace(
            go.Scatter(
                x=grid_x,
                y=grid_y,
                mode="lines",
                line=dict(color="yellow", width=0.5),
                opacity=0.6,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=(
            f"Projection at {first_angle:.3f}°"
            if first_angle is not None
            else "First projection"
        ),
        height=600,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    fig.update_xaxes(
        title_text="column", range=[-0.5, n_cols - 0.5], constrain="domain"
    )
    fig.update_yaxes(
        title_text="row",
        range=[n_rows - 0.5, -0.5],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
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
    mbirjax_params = dict((config or {}).get("mbirjax_config", {}))

    mo.stop(
        not mbirjax_params,
        mo.md("*No `mbirjax_config` parameters found in the config.*"),
    )

    # scale factors default to 1 when absent from the loaded config
    mbirjax_params.setdefault("row_scale", 1.0)
    mbirjax_params.setdefault("col_scale", 1.0)

    def make_mbirjax_widget(name, value):
        if isinstance(value, bool):
            return mo.ui.checkbox(value=value, label=name)
        if isinstance(value, int):
            return mo.ui.number(value=value, step=1, label=name)
        if isinstance(value, float):
            return mo.ui.number(value=value, step=0.1, label=name)
        return mo.ui.text(value=str(value), label=name)

    excluded_mbirjax_params = {"verbose", "print_logs"}
    mbirjax_widgets = mo.ui.dictionary(
        {
            name: make_mbirjax_widget(name, value)
            for name, value in mbirjax_params.items()
            if name not in excluded_mbirjax_params
        }
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("### **⚙️ MBIRJAX parameters**"),
                    mo.md(
                        '<a href="https://mbirjax.readthedocs.io/en/latest/'
                        'usr_parameters.html#positivity-flag" target="_blank" '
                        'title="MBIRJAX parameters documentation">🌐</a>'
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            *mbirjax_widgets.elements.values(),
        ]
    ).style(
        {
            "background-color": "#eef2f7",
            "color": "#1a1a1a",
            "padding": "1rem",
            "border-radius": "8px",
            "border": "1px solid #c5d0dd",
            "margin-top": "2rem",
        }
    )
    return (mbirjax_widgets,)


@app.cell
def _(first_image, mo):
    mo.stop(first_image is None)

    evaluate_reconstruction_button = mo.ui.run_button(
        label="Evaluate CT reconstruction of selected slices",
        kind="success",
        full_width=True,
    )
    evaluate_reconstruction_button
    return (evaluate_reconstruction_button,)


@app.cell
def _(
    angles_deg,
    bottom_line_slider,
    config,
    evaluate_reconstruction_button,
    mbirjax_widgets,
    mo,
    normalized_images_log,
    perform_tilt_switch,
    tilt_slider,
    top_line_slider,
    z_range_slider,
):
    mo.stop(
        not evaluate_reconstruction_button.value,
        mo.md("*Click the button above to evaluate the CT reconstruction.*"),
    )

    # parameters recovered from the widgets
    reconstruction_parameters = {
        "top_slice": top_line_slider.value,
        "bottom_slice": bottom_line_slider.value,
        "z_range": z_range_slider.value,
        "tilt": tilt_slider.value,
        "perform_tilt": perform_tilt_switch.value,
        "mbirjax_config": dict(mbirjax_widgets.value),
    }

    # data recovered from the selected HDF5 file
    reconstruction_config = config  # metadata/config
    reconstruction_data = normalized_images_log  # 3D stack (n_angles, rows, cols)
    reconstruction_angles = angles_deg  # projection angles (deg)

    mbirjax_lines = "\n".join(
        f"- **{name}:** {value}"
        for name, value in reconstruction_parameters["mbirjax_config"].items()
    )

    mo.vstack(
        [
            mo.md("### **🚀 Ready to evaluate CT reconstruction**"),
            mo.md(
                f"""
                - **top / bottom slice:** {reconstruction_parameters["top_slice"]} / {reconstruction_parameters["bottom_slice"]}
                - **z range:** {reconstruction_parameters["z_range"]}
                - **tilt (°):** {reconstruction_parameters["tilt"]}
                - **config:** {"loaded" if reconstruction_config is not None else "missing"}
                - **3D data:** {reconstruction_data.shape if reconstruction_data is not None else "missing"}
                - **angles:** {len(reconstruction_angles) if reconstruction_angles is not None else "missing"}
                - **size of input data for reconstruction:** {reconstruction_data.shape if reconstruction_data is not None else "missing"}
                """
            ),
            mo.md("\n**mbirjax parameters:**\n" + mbirjax_lines).style(
                {"margin-top": "2rem"}
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
    
    return (
        reconstruction_angles,
        reconstruction_config,
        reconstruction_data,
        reconstruction_parameters,
    )


@app.cell
def _(mo, 
      evaluate_reconstruction_button, 
      reconstruction_angles, 
      reconstruction_config, 
      reconstruction_data, 
      reconstruction_parameters,
      np):
    mo.stop(
        not evaluate_reconstruction_button.value,
    )

    from __code.marimo.mbirjax_reconstruction_evaluation import MbirjaxReconstructionEvaluation
    reconstruction_evaluation = MbirjaxReconstructionEvaluation(
        data=reconstruction_data,
        list_angles_deg=reconstruction_angles,
        reconstruction_parameters=reconstruction_parameters,
        )    
    top_reconstruction_array, bottom_reconstruction_array = reconstruction_evaluation.evaluate()
    
    print(f"{np.shape(top_reconstruction_array)=}")
    print(f"{np.shape(bottom_reconstruction_array)=}")

if __name__ == "__main__":
    app.run()
