import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import os
    import copy

    return copy, json, mo


@app.cell
def _(mo):
    mo.md(r"""
    # CT Reconstruction Config Editor

    Select a JSON configuration file, edit any parameter, then save it back.
    """)
    return


@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(
        initial_path="/SNS/VENUS",
        filetypes=[".json"],
        multiple=False,
        label="Select a JSON config file",
    )
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, json, mo):
    config = None
    config_path = None
    load_error = None

    if file_browser.value:
        config_path = file_browser.value[0].path
        try:
            raw = open(config_path).read()
            parsed = json.loads(raw)
            # The file may store a JSON-encoded string (double-serialised)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            config = parsed
        except Exception as e:
            load_error = str(e)

    if load_error:
        mo.stop(True, mo.callout(mo.md(f"**Could not load file:** {load_error}"), kind="danger"))

    if config is None:
        mo.stop(True, mo.callout(mo.md("No file selected yet."), kind="neutral"))

    mo.callout(mo.md(f"Loaded **{config_path}**"), kind="success")
    return config, config_path


@app.cell
def _(config, mo):
    # ── helpers ──────────────────────────────────────────────────────────────

    SKIP_KEYS = {"list_of_angles"}          # too long to display usefully
    LIST_PREVIEW_KEYS = {"list_of_slices_to_reconstruct",
                         "list_of_sample_runs", "list_of_ob_runs",
                         "list_of_sample_frame_number", "list_of_ob_frame_number",
                         "list_of_sample_pc", "list_of_ob_pc",
                         "range_of_tof_to_combine"}

    def make_widget(key, value):
        """Return an appropriate marimo UI widget for a config value."""
        label = key.replace("_", " ").title()

        if isinstance(value, bool):
            return mo.ui.switch(value=value, label=label)

        if isinstance(value, int):
            lo = min(0, value - abs(value) * 2)
            hi = max(value + abs(value) * 2, 10)
            return mo.ui.number(start=lo, stop=hi, step=1, value=value, label=label)

        if isinstance(value, float):
            lo = min(0.0, value - abs(value) * 2)
            hi = max(value + abs(value) * 2, 1.0)
            return mo.ui.number(start=lo, stop=hi, step=0.01, value=value, label=label)

        if isinstance(value, list):
            # Render as editable text (JSON array)
            import json as _json
            return mo.ui.text_area(value=_json.dumps(value), label=label, rows=2)

        # str / None / anything else
        return mo.ui.text(value=str(value) if value is not None else "", label=label)


    def section(title, fields_dict):
        """Wrap a group of widgets in a titled card."""
        return mo.vstack([
            mo.md(f"### {title}"),
            mo.vstack(list(fields_dict.values())),
            mo.md("---"),
        ])


    # ── build top-level flat widgets ─────────────────────────────────────────
    NESTED_KEYS = {"image_size", "crop_region", "normalization_roi",
                   "svmbir_config", "mbirjax_config",
                   "remove_stripe_fw_options", "remove_stripe_ti_options",
                   "remove_stripe_sf_options", "remove_stripe_based_sorting_options",
                   "remove_stripe_based_filtering_options",
                   "remove_stripe_based_fitting_options",
                   "remove_large_stripe_options", "remove_dead_stripe_options",
                   "remove_all_stripe_options",
                   "remove_stripe_based_interpolation_options",
                   "histogram_cleaning_settings",
                   "top_folder"}

    flat_widgets   = {}   # key -> widget
    nested_widgets = {}   # section_key -> {sub_key -> widget}

    for _k, _v in config.items():
        if _k in SKIP_KEYS:
            continue
        if _k in NESTED_KEYS and isinstance(_v, dict):
            nested_widgets[_k] = {sk: make_widget(sk, sv) for sk, sv in _v.items()}
        elif _k in LIST_PREVIEW_KEYS and isinstance(_v, list):
            import json as _j
            flat_widgets[_k] = mo.ui.text_area(
                value=_j.dumps(_v, indent=2),
                label=_k.replace("_", " ").title(),
                rows=4,
            )
        else:
            flat_widgets[_k] = make_widget(_k, _v)

    flat_widgets, nested_widgets
    return flat_widgets, nested_widgets, section


@app.cell
def _(flat_widgets, mo, nested_widgets, section):
    # ── render the form ───────────────────────────────────────────────────────

    def render_nested(key, sub_widgets):
        title = key.replace("_", " ").title()
        return section(title, sub_widgets)

    panels = []

    # General / top-level parameters first
    general_keys = [k for k in flat_widgets if k not in
                    {"output_folder", "reconstructed_output_folder",
                     "projections_pre_processing_folder",
                     "raw_data_base_folder"}]
    path_keys    = ["output_folder", "reconstructed_output_folder",
                    "projections_pre_processing_folder", "raw_data_base_folder"]

    if general_keys:
        panels.append(section("General Parameters",
                               {k: flat_widgets[k] for k in general_keys if k in flat_widgets}))

    for nk, nw in nested_widgets.items():
        panels.append(render_nested(nk, nw))

    if any(k in flat_widgets for k in path_keys):
        panels.append(section("Paths & Folders",
                               {k: flat_widgets[k] for k in path_keys if k in flat_widgets}))

    mo.vstack(panels)
    return


@app.cell
def _(mo):
    # ── save button ───────────────────────────────────────────────────────────

    save_btn = mo.ui.button(label="💾  Save config", kind="success")
    save_btn
    return (save_btn,)


@app.cell
def _(
    config,
    config_path,
    copy,
    flat_widgets,
    json,
    mo,
    nested_widgets,
    save_btn,
):
    save_status = mo.md("")

    if save_btn.value:
        updated = copy.deepcopy(config)

        # apply flat widget values
        for _key, _widget in flat_widgets.items():
            raw_val = _widget.value
            orig    = config.get(_key)
            if isinstance(orig, list):
                try:
                    raw_val = json.loads(raw_val)
                except Exception:
                    pass   # keep as string if parse fails
            elif isinstance(orig, bool):
                raw_val = bool(raw_val)
            elif isinstance(orig, int):
                raw_val = int(raw_val)
            elif isinstance(orig, float):
                raw_val = float(raw_val)
            updated[_key] = raw_val

        # apply nested widget values
        for _section_key, _sub_widgets in nested_widgets.items():
            for _sk, _sw in _sub_widgets.items():
                raw_val = _sw.value
                orig    = config.get(_section_key, {}).get(_sk)
                if isinstance(orig, bool):
                    raw_val = bool(raw_val)
                elif isinstance(orig, int):
                    raw_val = int(raw_val)
                elif isinstance(orig, float):
                    raw_val = float(raw_val)
                elif isinstance(orig, list):
                    try:
                        raw_val = json.loads(raw_val)
                    except Exception:
                        pass
                updated[_section_key][_sk] = raw_val

        try:
            with open(config_path, "w") as _f:
                json.dump(updated, _f, indent=2)
            save_status = mo.callout(
                mo.md(f"✅ Saved successfully to **{config_path}**"),
                kind="success",
            )
        except Exception as _e:
            save_status = mo.callout(
                mo.md(f"❌ Save failed: {_e}"),
                kind="danger",
            )

    save_status
    return


if __name__ == "__main__":
    app.run()
