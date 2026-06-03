#!/bin/bash
cd "$(dirname "$0")"
XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run marimo edit notebooks/optimize_mbirjax_parameters.py
