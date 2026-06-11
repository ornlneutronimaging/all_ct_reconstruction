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
        _log_dir, f"optimize_svmbir_parameters_{_user_id}.log"
    )

    # a dedicated, non-propagating logger so it is independent of the root logger
    # (which the reconstruction-evaluation module reconfigures on import and would
    # otherwise redirect or duplicate these messages)
    logger = _logging.getLogger("optimize_svmbir_parameters")
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
    logger.info("*** optimize_svmbir_parameters session started ***")
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
def _(
    angles_deg,
    bottom_line_slider,
    colormap_selector,
    go,
    svmbir_widgets,
    mo,
    mpl_colormap_to_plotly,
    normalized_images_log,
    np,
    perform_tilt_switch,
    projection_selector,
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

    from __code.config import MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH
    band_half = int(MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH / 2)

    first_image = normalized_images_log[0]

    # find index closest to 180°
    if angles_deg is not None and len(angles_deg):
        idx_180 = int(np.argmin(np.abs(angles_deg - 180.0)))
    else:
        idx_180 = 0

    sel = projection_selector.value
    if sel == "180°":
        selected_indices = [idx_180]
    elif sel == "0° and 180°":
        selected_indices = [0, idx_180]
    else:  # "0°"
        selected_indices = [0]

    vmin, vmax = z_range_slider.value
    tilt_angle = tilt_slider.value
    perform_tilt = perform_tilt_switch.value
    n_rows, n_cols = first_image.shape
    center_column = first_image.shape[1] / 2
    center_offset = svmbir_widgets.value.get("center_offset", 0)

    # subsample the displayed heatmap when the image is large so the preview
    # stays responsive; pass original-coordinate x/y arrays so the axes keep
    # showing the full array size and the overlays remain aligned
    downsample = 10 if (n_rows > 1000 or n_cols > 1000) else 1

    def _heatmap_data(img):
        if downsample > 1:
            return dict(
                z=img[::downsample, ::downsample],
                x=np.arange(0, n_cols, downsample),
                y=np.arange(0, n_rows, downsample),
            )
        return dict(z=img)

    def _get_display_image(idx):
        raw_img = normalized_images_log[idx]
        return (
            rotate(raw_img, angle=tilt_angle, reshape=False, order=1, mode="nearest")
            if perform_tilt and tilt_angle != 0
            else raw_img
        )

    def _add_overlays(f):
        # the reconstructed region is only MARIMO_SVMBIR_TEST_RECONSTRUCTION_WIDTH px
        # tall, so on a full detector image it is a thin sliver; draw the band
        # with a stronger fill plus solid edge lines at its top/bottom so the
        # region used for reconstruction clearly stands out
        f.add_hrect(
            y0=top_line_slider.value - band_half,
            y1=top_line_slider.value + band_half,
            fillcolor="red",
            opacity=0.45,
            line=dict(color="red", width=1.5),
        )
        f.add_hline(y=top_line_slider.value, line_color="red", line_width=2.0)
        f.add_hrect(
            y0=bottom_line_slider.value - band_half,
            y1=bottom_line_slider.value + band_half,
            fillcolor="cyan",
            opacity=0.45,
            line=dict(color="cyan", width=1.5),
        )
        f.add_hline(y=bottom_line_slider.value, line_color="cyan", line_width=2.0)
        if center_offset is not None:
            f.add_vline(
                x=center_column + center_offset,
                line_color="white",
                line_dash="dash",
                line_width=1.0,
            )
        if show_grid_toggle.value:
            grid_angle = 0.0 if perform_tilt else tilt_angle
            _center = np.array([n_cols / 2.0, n_rows / 2.0])
            theta = np.deg2rad(-grid_angle)
            along = np.array([np.cos(theta), -np.sin(theta)])
            across = np.array([np.sin(theta), np.cos(theta)])
            diag = float(np.hypot(n_rows, n_cols))
            spacing = max(n_rows, n_cols) / 10.0
            n_lines = int(diag / spacing) + 1
            grid_x = []
            grid_y = []
            for k in range(-n_lines, n_lines + 1):
                for direction, offset_dir in ((along, across), (across, along)):
                    base = _center + k * spacing * offset_dir
                    p0 = base - diag * direction
                    p1 = base + diag * direction
                    grid_x += [p0[0], p1[0], None]
                    grid_y += [p0[1], p1[1], None]
            f.add_trace(
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

    def _finalize_fig(f, title):
        f.update_layout(title=title, height=600, margin=dict(l=60, r=20, t=50, b=50))
        f.update_xaxes(
            title_text="column", range=[-0.5, n_cols - 0.5], constrain="domain"
        )
        f.update_yaxes(
            title_text="row",
            range=[n_rows - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )

    def _angle_label(idx):
        if angles_deg is not None and len(angles_deg) > idx:
            return f"{float(angles_deg[idx]):.3f}°"
        return f"#{idx}"

    if sel == "0° and 180°":
        # overlay: average both images into a single heatmap
        avg_img = (_get_display_image(0) + _get_display_image(idx_180)) / 2.0
        fig = go.Figure(
            go.Heatmap(
                **_heatmap_data(avg_img),
                colorscale=mpl_colormap_to_plotly(colormap_selector.value),
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="intensity"),
            )
        )
        _add_overlays(fig)
        _finalize_fig(
            fig,
            f"Overlay: {_angle_label(0)} and {_angle_label(idx_180)} (averaged)",
        )
        figs = [fig]
    else:
        idx = selected_indices[0]
        fig = go.Figure(
            go.Heatmap(
                **_heatmap_data(_get_display_image(idx)),
                colorscale=mpl_colormap_to_plotly(colormap_selector.value),
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="intensity"),
            )
        )
        _add_overlays(fig)
        _finalize_fig(fig, f"Projection at {_angle_label(idx)}")
        figs = [fig]

    mo.vstack(
        [
            projection_selector,
            mo.hstack(
                [
                    *figs,
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
    projection_selector = mo.ui.radio(
        options=["0°", "180°", "0° and 180°"],
        value="0°",
        label="**Projection:**",
        inline=True,
    )
    mo.vstack(
        [
            top_line_slider,
            bottom_line_slider,
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
        projection_selector,
        show_grid_toggle,
        tilt_slider,
        top_line_slider,
        z_range_slider,
    )


@app.cell
def _(mo):
    # password gate for the Advanced parameters section; defined in its own cell
    # so typing into it re-runs the rendering cell (but not this definition)
    advanced_password = mo.ui.text(
        kind="password",
        placeholder="Enter password to unlock",
        label="🔒 **Advanced** parameters password:",
    )
    return (advanced_password,)


@app.cell
def _(advanced_password, config, mo, normalized_images_log):
    svmbir_params = dict((config or {}).get("svmbir_config", {}))

    mo.stop(
        not svmbir_params,
        mo.md("*No `svmbir_config` parameters found in the config.*"),
    )

    # top_slice / bottom_slice are driven by the line sliders above, not by these
    # parameter widgets; drop them so they are not rendered here
    svmbir_params.pop("top_slice", None)
    svmbir_params.pop("bottom_slice", None)

    # svmbir takes the center of rotation as a pixel offset from the detector
    # center; it is not part of svmbir_config, so seed it from the top-level
    # center_of_rotation when present, otherwise default to 0
    if "center_offset" not in svmbir_params:
        _center_of_rotation = (config or {}).get("center_of_rotation")
        if (
            _center_of_rotation is not None
            and normalized_images_log is not None
            and len(normalized_images_log)
        ):
            _width = normalized_images_log[0].shape[1]
            svmbir_params["center_offset"] = float(-(_width // 2 - _center_of_rotation))
        else:
            svmbir_params["center_offset"] = 0.0

    from __code.utilities.configuration_file import MinMaxRange

    # parameters rendered as sliders, with (range, step, is_integer)
    slider_specs = {
        "max_iterations": (MinMaxRange(min=10, max=100, default=20), 1, True),
        "sharpness": (MinMaxRange(min=-1, max=3, default=0), 0.1, False),
        "snr_db": (MinMaxRange(min=25, max=40, default=30), 1, False),
        "max_resolutions": (MinMaxRange(min=0, max=4, default=3), 1, True),
    }

    # always start maximum iterations at its default, ignoring any (typically
    # much larger) value stored in the loaded config
    svmbir_params["max_iterations"] = slider_specs["max_iterations"][0].default

    # display labels overriding the raw parameter key for the widget label
    svmbir_param_labels = {
        "max_iterations": "maximum iterations",
        "center_offset": "center of rotation offset from center",
        "snr_db": "signal noise ratio",
        "max_resolutions": "maximum resolutions",
    }

    def _display_label(name):
        return svmbir_param_labels.get(name, name)

    # center_offset is an offset from the image center, bounded to +/- half the
    # image width (fall back to a sane default if no image is loaded)
    if normalized_images_log is not None and len(normalized_images_log):
        offset_limit = float(normalized_images_log[0].shape[1]) / 2
    else:
        offset_limit = 256.0

    def make_svmbir_widget(name, value):
        if name == "center_offset":
            current = float(value) if value is not None else 0.0
            current = max(-offset_limit, min(offset_limit, current))
            return mo.ui.number(
                start=-offset_limit,
                stop=offset_limit,
                step=0.01,
                value=round(current, 2),
                label=_display_label(name),
            )
        if name in slider_specs:
            rng, step, is_int = slider_specs[name]
            current = float(value) if value is not None else float(rng.default)
            current = max(rng.min, min(rng.max, current))
            if is_int:
                return mo.ui.slider(
                    start=int(rng.min),
                    stop=int(rng.max),
                    step=int(step),
                    value=int(round(current)),
                    label=_display_label(name),
                    show_value=True,
                    full_width=True,
                )
            return mo.ui.slider(
                start=rng.min,
                stop=rng.max,
                step=step,
                value=current,
                label=_display_label(name),
                show_value=True,
                full_width=True,
            )
        if isinstance(value, bool):
            return mo.ui.checkbox(value=value, label=_display_label(name))
        if isinstance(value, int):
            return mo.ui.number(value=value, step=1, label=_display_label(name))
        if isinstance(value, float):
            return mo.ui.number(value=value, step=0.1, label=_display_label(name))
        return mo.ui.text(value=str(value), label=_display_label(name))

    excluded_svmbir_params = {"verbose", "print_logs"}
    svmbir_widgets = mo.ui.dictionary(
        {
            name: make_svmbir_widget(name, value)
            for name, value in svmbir_params.items()
            if name not in excluded_svmbir_params
        }
    )

    # short explanation of each svmbir parameter, shown as a hover tooltip on
    # the ℹ️ icon next to the corresponding widget
    svmbir_param_descriptions = {
        "positivity": (
            "Enforce a positivity constraint: all reconstructed voxel values are "
            "forced to be greater than or equal to zero."
        ),
        "max_iterations": (
            "Maximum number of iterations of the iterative reconstruction "
            "algorithm. More iterations improve convergence but increase the "
            "reconstruction time."
        ),
        "sharpness": (
            "Controls the sharpness of the reconstruction. Larger (positive) "
            "values produce sharper, more detailed images; smaller (negative) "
            "values produce smoother images."
        ),
        "snr_db": (
            "Assumed signal-to-noise ratio of the data, in decibels. Larger "
            "values yield sharper reconstructions but can amplify noise."
        ),
        "max_resolutions": (
            "Maximum number of multi-resolution levels used to accelerate "
            "convergence. Higher values begin the reconstruction on a coarser "
            "grid; 0 disables the multi-resolution scheme."
        ),
        "center_offset": (
            "Offset of the center of rotation from the center of the detector "
            "image, in pixels. Positive values shift the center to the right."
        ),
    }

    def _info_icon(name):
        # ℹ️ icon whose native browser tooltip (title attribute) shows the
        # parameter's meaning on hover
        description = svmbir_param_descriptions.get(
            name, "No description available."
        )
        return mo.Html(
            f'<span title="{description}" '
            'style="cursor: help; font-size: 1.1rem;">ℹ️</span>'
        )

    # render each widget with an info icon to its left; center_offset keeps an
    # extra inline label spelling out the center-of-rotation offset. The
    # parameters are split into a "General" section (sharpness) and an
    # "Advanced" section (everything else)
    general_param_names = {"sharpness"}
    general_rows = []
    advanced_rows = []
    for name, element in svmbir_widgets.elements.items():
        row = mo.hstack(
            [_info_icon(name), element],
            justify="start",
            align="center",
            gap=0.5,
        )
        if name in general_param_names:
            general_rows.append(row)
        else:
            advanced_rows.append(row)

    # the Advanced section is hidden until the correct password is entered; the
    # widgets themselves are always defined above, only their display is gated
    _advanced_unlocked = advanced_password.value == "venus"

    _general_section = mo.vstack(
        [
            mo.md("#### **General**"),
            *general_rows,
        ]
    )

    if _advanced_unlocked:
        _advanced_body = mo.vstack(advanced_rows)
    else:
        _advanced_body = mo.md(
            "🔒 *Enter the password above to unlock the advanced parameters.*"
        )

    _advanced_section = mo.vstack(
        [
            mo.md("#### **🔧 Advanced**"),
            advanced_password,
            _advanced_body,
        ]
    ).style(
        {
            "background-color": "#f7f0e6",
            "padding": "1rem",
            "border-radius": "8px",
            "border": "1px solid #ddc9a3",
            "margin-top": "1rem",
        }
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("### **⚙️ SVMBIR parameters**"),
                    mo.md(
                        '<a href="https://svmbir.readthedocs.io/en/latest/'
                        'parameters.html" target="_blank" '
                        'title="SVMBIR parameters documentation">🌐</a>'
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            _general_section,
            _advanced_section,
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
    return (svmbir_widgets,)


@app.cell
def _(mo):
    get_show_log, set_show_log = mo.state(False)
    return get_show_log, set_show_log


@app.cell
def _(mo):
    # accumulated reconstructions; each click appends one entry, rendered as a row
    get_reconstruction_history, set_reconstruction_history = mo.state([])
    return get_reconstruction_history, set_reconstruction_history


@app.cell
def _(mo):
    # reconstruction_parameters dict chosen by the user via "Use this configuration"
    get_selected_config, set_selected_config = mo.state(None)
    return get_selected_config, set_selected_config


@app.cell
def _(mo):
    # True while a reconstruction runs on a background thread, so the rest of the
    # app stays interactive; drives the progress indicator below the run button
    get_is_reconstructing, set_is_reconstructing = mo.state(False)
    return get_is_reconstructing, set_is_reconstructing


@app.cell
def _(mo):
    # full reconstruction volumes (top/bottom) from the most recent run; reused
    # as init_recon for the next run so each reconstruction builds on the last
    get_last_full_reconstruction, set_last_full_reconstruction = mo.state(None)
    return get_last_full_reconstruction, set_last_full_reconstruction


@app.cell
def _(
    first_image,
    get_is_reconstructing,
    get_last_full_reconstruction,
    mo,
    set_last_full_reconstruction,
    set_show_log,
):
    mo.stop(first_image is None)

    # disabled while a reconstruction is running so a second one can't be
    # launched mid-run; re-enabled when the background thread clears the flag
    _reconstructing = get_is_reconstructing()
    evaluate_reconstruction_button = mo.ui.run_button(
        label="Click to evaluate CT reconstruction of selected slices",
        kind="success",
        full_width=True,
        disabled=_reconstructing,
        tooltip="Reconstruction in progress ..." if _reconstructing else None,
    )

    # clears the stored reconstruction so the next run starts from scratch
    # instead of warm-starting (init_recon) from the previous result; disabled
    # when there is nothing stored or a reconstruction is in progress
    _has_previous = get_last_full_reconstruction() is not None
    reset_reconstruction_button = mo.ui.button(
        label="♻️ reset",
        kind="warn",
        on_change=lambda _: set_last_full_reconstruction(None),
        disabled=_reconstructing or not _has_previous,
        tooltip=(
            "Reconstruction in progress ..."
            if _reconstructing
            else "Forget the previous reconstruction so the next run starts fresh"
            if _has_previous
            else "No previous reconstruction to reset"
        ),
    )

    display_log_button = mo.ui.button(
        label="📖 display log",
        kind="neutral",
        on_change=lambda _: set_show_log(True),
    )
    hide_log_button = mo.ui.button(
        label="📕 hide log",
        kind="neutral",
        on_change=lambda _: set_show_log(False),
    )
    return (
        display_log_button,
        evaluate_reconstruction_button,
        hide_log_button,
        reset_reconstruction_button,
    )


@app.cell
def _(get_reconstruction_history, mo):
    # one "use this configuration" button per reconstruction in the history
    use_configuration_buttons = mo.ui.array(
        [
            mo.ui.run_button(
                label="👉 Use this configuration",
                kind="success",
                full_width=True,
            )
            for _ in get_reconstruction_history()
        ]
    )
    return (use_configuration_buttons,)


@app.cell
def _(get_reconstruction_history, set_selected_config, use_configuration_buttons):
    # detect which "Use this configuration" button was clicked and record the
    # full reconstruction_parameters for that history entry
    _history = get_reconstruction_history()
    for _i, _btn in enumerate(use_configuration_buttons):
        if _btn.value and _i < len(_history):
            set_selected_config(_history[_i]["reconstruction_parameters"])
    return


@app.cell
def _(
    display_log_button,
    evaluate_reconstruction_button,
    get_show_log,
    hide_log_button,
    mo,
    reset_reconstruction_button,
):
    _log_button = hide_log_button if get_show_log() else display_log_button

    mo.hstack(
        [
            evaluate_reconstruction_button,
            reset_reconstruction_button.style({"width": "120px"}),
            _log_button.style({"width": "200px"}),
        ],
        justify="space-between",
        align="center",
        gap=0.75,
    )
    return


@app.cell
def _(get_is_reconstructing, mo):
    # shown while the background reconstruction thread is running; the flag is
    # cleared by the thread on completion, which removes this indicator
    mo.stop(not get_is_reconstructing())

    mo.hstack(
        [
            mo.status.spinner(title="Reconstruction in progress ..."),
            mo.md("*You can keep adjusting parameters while this runs.*"),
        ],
        justify="start",
        align="center",
        gap=0.75,
    )
    return


@app.cell
def _(get_show_log, mo, os):
    import getpass
    import html

    mo.stop(not get_show_log())

    _user_id = getpass.getuser()
    _log_path = (
        f"/SNS/VENUS/shared/log/svmbir_reconstruction_evaluation_{_user_id}.log"
    )

    if os.path.exists(_log_path):
        with open(_log_path, "r") as _f:
            _log_content = _f.read()
        _log_body = mo.Html(
            "<div style='display:flex; flex-direction:column-reverse; "
            "max-height:400px; overflow-y:auto;'>"
            "<pre style='white-space:pre-wrap; word-break:break-word; "
            "margin:0; font-family:monospace; font-size:0.8rem;'>"
            f"{html.escape(_log_content) or '(log file is empty)'}</pre>"
            "</div>"
        )
    else:
        _log_body = mo.md(f"*Log file not found: `{_log_path}`*")

    mo.vstack(
        [
            mo.md(f"### **📜 Log preview** — `{_log_path}`"),
            _log_body,
        ]
    ).style(
        {
            "background-color": "#eef2f7",
            "color": "#1a1a1a",
            "padding": "1rem",
            "border-radius": "8px",
            "border": "1px solid #c5d0dd",
            "margin-top": "1rem",
        }
    )
    return


@app.cell
def _(
    angles_deg,
    bottom_line_slider,
    config,
    evaluate_reconstruction_button,
    svmbir_widgets,
    mo,
    normalized_images_log,
    perform_tilt_switch,
    tilt_slider,
    top_line_slider,
    z_range_slider,
):
    mo.stop(
        not evaluate_reconstruction_button.value,
    )

    # parameters recovered from the widgets
    _svmbir_config = dict(svmbir_widgets.value)
    reconstruction_parameters = {
        "top_slice": top_line_slider.value,
        "bottom_slice": bottom_line_slider.value,
        "z_range": z_range_slider.value,
        "tilt": tilt_slider.value,
        "perform_tilt": perform_tilt_switch.value,
        "svmbir_config": _svmbir_config,
    }

    # data recovered from the selected HDF5 file
    reconstruction_config = config  # metadata/config
    reconstruction_data = normalized_images_log  # 3D stack (n_angles, rows, cols)
    reconstruction_angles = angles_deg  # projection angles (deg)

    svmbir_lines = "\n".join(
        f"- **{name}:** {value}"
        for name, value in reconstruction_parameters["svmbir_config"].items()
    )

    mo.vstack(
        [
            mo.md("### **🚀 CT reconstruction will be evaluated using those parameters**"),
            mo.hstack(
                [
                    mo.md(
                        "**General parameters:**\n"
                        f"""
                - **top / bottom slice:** {reconstruction_parameters["top_slice"]} / {reconstruction_parameters["bottom_slice"]}
                - **tilt (°):** {reconstruction_parameters["tilt"]}
                - **3D data:** {reconstruction_data.shape if reconstruction_data is not None else "missing"}
                - **angles:** {len(reconstruction_angles) if reconstruction_angles is not None else "missing"}
                - **size of input data for reconstruction:** {reconstruction_data.shape if reconstruction_data is not None else "missing"}
                """
                    ),
                    mo.md("**svmbir parameters:**\n" + svmbir_lines),
                ],
                widths="equal",
                gap=2,
                align="start",
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
def _(
    evaluate_reconstruction_button,
    get_last_full_reconstruction,
    mo,
    reconstruction_angles,
    reconstruction_data,
    reconstruction_parameters,
    set_is_reconstructing,
    set_last_full_reconstruction,
    set_reconstruction_history,
    logger,
):
    mo.stop(
        not evaluate_reconstruction_button.value,
    )

    from __code.marimo.svmbir_reconstruction_evaluation import SvmbirReconstructionEvaluation

    # Use default argument values to capture inputs immediately at definition
    # time. This avoids closure over _-prefixed cell-level variables, which
    # marimo mangles (e.g. _parameters -> _cell_Vxnm_parameters) making them
    # unavailable when the thread actually executes.
    def _run_reconstruction(
        snap_data=reconstruction_data,
        snap_angles=reconstruction_angles,
        snap_parameters=reconstruction_parameters,
        snap_init_recon=get_last_full_reconstruction(),
    ):

        logger.info("Starting svmbir reconstruction...")
        logger.info(f"Data shape: {snap_data.shape if snap_data is not None else 'missing'}")
        logger.info(f"Angles: {len(snap_angles) if snap_angles is not None else 'missing'}")
        logger.info(f"Parameters: {snap_parameters}")
        logger.info(f"Initial reconstruction: {snap_init_recon is not None}")

        # runs on a mo.Thread; svmbir's C extension releases the GIL during
        # compute, so the rest of the app stays interactive while this runs
        try:
            # seed this run with the previous reconstruction (if any); the
            # evaluation ignores it when the recon grid no longer matches
            evaluation = SvmbirReconstructionEvaluation(
                data=snap_data,
                list_angles_deg=snap_angles,
                reconstruction_parameters=snap_parameters,
                init_recon=snap_init_recon,
            )
            top_slice, bottom_slice, top_time, bottom_time = evaluation.evaluate()
            logger.info(
                f"Reconstruction finished | top_time={top_time:.2f}s, "
                f"bottom_time={bottom_time:.2f}s"
            )

            # remember the full reconstruction volumes so the next run can reuse
            # them as init_recon
            set_last_full_reconstruction(
                {
                    "top": evaluation.top_full_reconstruction,
                    "bottom": evaluation.bottom_full_reconstruction,
                }
            )

            # Store a presampled copy for the preview rather than the full
            # resolution slice. The slices are only ever shown as downsampled
            # heatmaps, and the reconstruction-history cell renders *every* row
            # in a single output, so each added reconstruction enlarges that
            # cell's serialized output until marimo rejects it as "too large".
            # Presampling to a small grid (and rounding the values, which
            # shortens the JSON text plotly emits for the z array) keeps each
            # row tiny so many reconstructions fit. The original shape is
            # recorded so the preview axes still reflect the true dimensions.
            import numpy as _np
            _preview_max_dim = 100

            def _subsample_for_preview(_arr):
                _arr = _np.asarray(_np.squeeze(_arr), dtype=_np.float32)
                _r, _c = _arr.shape
                _stride = max(1, int(_np.ceil(max(_r, _c) / _preview_max_dim)))
                _small = _arr[::_stride, ::_stride]
                # round to ~4 significant figures relative to the data range so
                # the serialized heatmap stays compact without visible change
                _scale = float(_np.nanmax(_np.abs(_small))) if _small.size else 0.0
                if _scale > 0:
                    _decimals = max(0, 4 - 1 - int(_np.floor(_np.log10(_scale))))
                    _small = _np.round(_small, _decimals)
                return {
                    "data": _small,
                    "shape": (_r, _c),
                    "stride": _stride,
                }

            new_entry = {
                "top": _subsample_for_preview(top_slice),
                "bottom": _subsample_for_preview(bottom_slice),
                "reconstruction_parameters": snap_parameters,
                "svmbir_config": dict(snap_parameters["svmbir_config"]),
                "top_reconstruction_time": top_time,
                "bottom_reconstruction_time": bottom_time,
            }

            # state set from a mo.Thread re-runs the dependent (display) cells,
            # appending this reconstruction as a new row below the others
            set_reconstruction_history(
                lambda prev, entry=new_entry: prev + [entry]
            )
        except Exception:
            logger.exception("Reconstruction failed")
            raise
        finally:
            set_is_reconstructing(False)

    set_is_reconstructing(True)
    mo.Thread(target=_run_reconstruction).start()
    return


@app.cell
def _(
    colormap_selector,
    get_reconstruction_history,
    go,
    mo,
    mpl_colormap_to_plotly,
    np,
    use_configuration_buttons,
):
    _history = get_reconstruction_history()
    mo.stop(
        not _history,
    )

    def _central_slice_figure(slice_preview, title):
        # slice_preview is the subsampled record built at storage time:
        # {"data": downsampled 2D array, "shape": original (rows, cols),
        #  "stride": subsampling factor}. The downsampling happens once, when
        # the reconstruction is stored, so the history never holds full-size
        # arrays. Axes still use the original shape so they show the true size.
        _z_disp = np.asarray(slice_preview["data"], dtype=np.float32)
        _n_rows, _n_cols = slice_preview["shape"]
        _stride = slice_preview["stride"]
        _rows = np.arange(_n_rows)[::_stride]
        _cols = np.arange(_n_cols)[::_stride]
        _zmin = float(np.percentile(_z_disp, 1))
        _zmax = float(np.percentile(_z_disp, 99))
        _fig = go.Figure(
            go.Heatmap(
                z=_z_disp,
                x=_cols,
                y=_rows,
                zmin=_zmin,
                zmax=_zmax,
                colorscale=mpl_colormap_to_plotly(colormap_selector.value),
                colorbar=dict(title="intensity"),
            )
        )
        _fig.update_layout(
            title=title,
            height=450,
            margin=dict(l=50, r=20, t=50, b=50),
        )
        _fig.update_xaxes(
            title_text="column", range=[-0.5, _n_cols - 0.5], constrain="domain"
        )
        _fig.update_yaxes(
            title_text="row",
            range=[_n_rows - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )
        return _fig

    def _params_panel(entry, use_configuration_button):
        _svmbir_lines = "\n".join(
            f"- **{name}:** {value}" for name, value in entry["svmbir_config"].items()
        )
        _rp = entry.get("reconstruction_parameters", {})
        _tilt = _rp.get("tilt", 0.0)
        # the tilt is always applied before running the reduction, so there is
        # no need to annotate whether it was applied
        _tilt_line = f"**tilt (°):** {_tilt}"
        _time_lines = (
            "**reconstruction time:**<br>"
            f"**top:** {entry['top_reconstruction_time']:.2f} s<br>"
            f"**bottom:** {entry['bottom_reconstruction_time']:.2f} s"
        )
        return mo.vstack(
            [
                mo.md("**svmbir parameters used:**\n" + _svmbir_lines),
                mo.md(_tilt_line),
                mo.md(_time_lines),
                use_configuration_button,
            ],
            justify="space-between",
        ).style(
            {
                "background-color": "#eef2f7",
                "color": "#1a1a1a",
                "padding": "1rem",
                "border-radius": "8px",
                "border": "1px solid #c5d0dd",
                "min-width": "260px",
                "height": "100%",
            }
        )

    def _reconstruction_row(entry, use_configuration_button):
        return mo.hstack(
            [
                _central_slice_figure(entry["top"], "Central slice of top range"),
                _central_slice_figure(entry["bottom"], "Central slice of bottom range"),
                _params_panel(entry, use_configuration_button),
            ],
            widths=[3, 3, 2],
            gap=1,
            align="stretch",
        )

    mo.vstack(
        [
            _reconstruction_row(entry, use_configuration_buttons[_i])
            for _i, entry in enumerate(_history)
        ],
        gap=2,
    )
    return


@app.cell
def _(mo):
    # bottom-of-screen actions for the selected reconstruction configuration;
    # defined in their own cell so clicking one does not re-run the definition
    create_new_hdf5_button = mo.ui.run_button(
        label="Create new HDF5 with selected configuration",
        kind="success",
    )
    overwrite_hdf5_button = mo.ui.run_button(
        label="Overwrite current HDF5 with selected configuration",
        kind="warn",
    )
    return create_new_hdf5_button, overwrite_hdf5_button


@app.cell
def _(normalized_images_log):
    import copy as _copy_for_export

    def build_export_configuration(selected_config):
        # the notebook tunes the center of rotation as a pixel offset
        # (center_offset) inside svmbir_config, but the production svmbir
        # pipeline reads it from the top-level center_of_rotation. Convert it
        # back here and drop the helper key so the exported svmbir_config holds
        # only genuine svmbir parameters.
        cfg = _copy_for_export.deepcopy(selected_config)
        svmbir_config = dict(cfg.get("svmbir_config", {}))
        center_offset = svmbir_config.pop("center_offset", None)
        cfg["svmbir_config"] = svmbir_config
        if (
            center_offset is not None
            and normalized_images_log is not None
            and len(normalized_images_log)
        ):
            width = normalized_images_log[0].shape[1]
            cfg["center_of_rotation"] = int(round(width // 2 + center_offset))
        return cfg

    return (build_export_configuration,)


@app.cell
def _(
    create_new_hdf5_button,
    get_reconstruction_history,
    get_selected_config,
    mo,
    overwrite_hdf5_button,
):
    # only shown once at least one reconstruction has been evaluated
    mo.stop(
        not get_reconstruction_history(),
    )

    _cfg = get_selected_config()
    if _cfg is not None:
        _svmbir_lines = "\n".join(
            f"- **{name}:** {value}"
            for name, value in _cfg["svmbir_config"].items()
        )
        _selected_box = mo.vstack(
            [
                mo.md("### **✅ Selected configuration**"),
                mo.hstack(
                    [
                        mo.md(
                            "**General parameters:**\n"
                            f"- **top / bottom slice:** {_cfg['top_slice']} / {_cfg['bottom_slice']}\n"
                            f"- **tilt (°):** {_cfg['tilt']}\n"
                            f"- **perform tilt:** {_cfg['perform_tilt']}"
                        ),
                        mo.md("**svmbir parameters:**\n" + _svmbir_lines),
                    ],
                    widths="equal",
                    gap=2,
                    align="start",
                ),
            ]
        ).style(
            {
                "background-color": "#e8f5e9",
                "color": "#1a1a1a",
                "padding": "1rem",
                "border-radius": "8px",
                "border": "1px solid #81c784",
            }
        )
    else:
        _selected_box = mo.callout(
            mo.md("*No configuration selected yet — click **👉 Use this configuration** on a reconstruction above.*"),
            kind="info",
        )

    mo.vstack(
        [
            _selected_box,
            mo.hstack(
                [create_new_hdf5_button, overwrite_hdf5_button],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    build_export_configuration,
    create_new_hdf5_button,
    get_selected_config,
    mo,
    os,
    selected_hdf5_file,
):
    mo.stop(not create_new_hdf5_button.value)

    _cfg = get_selected_config()
    mo.stop(
        _cfg is None,
        mo.callout(
            mo.md("*No configuration selected — click **👉 Use this configuration** on a reconstruction first.*"),
            kind="warn",
        ),
    )

    from __code.marimo.export_new_configuration_to_hdf5 import ExportNewConfigurationToHDF5

    _base, _ext = os.path.splitext(selected_hdf5_file)
    _new_hdf5_path = _base + "_new_svmbir_config" + _ext

    ExportNewConfigurationToHDF5(
        new_configuration=build_export_configuration(_cfg),
        hdf5_file_path=_new_hdf5_path,
        source_hdf5_file_path=selected_hdf5_file,
        new_hdf5_flag=True,
    ).export()

    mo.callout(
        mo.md(f"New HDF5 file created: `{os.path.basename(_new_hdf5_path)}`"),
        kind="success",
    )
    return


@app.cell
def _(
    build_export_configuration,
    get_selected_config,
    mo,
    os,
    overwrite_hdf5_button,
    selected_hdf5_file,
):
    mo.stop(not overwrite_hdf5_button.value)

    _cfg = get_selected_config()
    mo.stop(
        _cfg is None,
        mo.callout(
            mo.md("*No configuration selected — click **👉 Use this configuration** on a reconstruction first.*"),
            kind="warn",
        ),
    )

    from __code.marimo.export_new_configuration_to_hdf5 import ExportNewConfigurationToHDF5 as _ExportNewConfigurationToHDF5

    _ExportNewConfigurationToHDF5(
        new_configuration=build_export_configuration(_cfg),
        hdf5_file_path=selected_hdf5_file,
        new_hdf5_flag=False,
    ).export()

    mo.callout(
        mo.md(f"HDF5 configuration updated in-place: `{os.path.basename(selected_hdf5_file)}`"),
        kind="warn",
    )
    return


if __name__ == "__main__":
    app.run()
