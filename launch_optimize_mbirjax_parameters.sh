#!/bin/bash
cd "$(dirname "$0")"
XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run marimo run notebooks/marimo_optimize_mbirjax_parameters.py
