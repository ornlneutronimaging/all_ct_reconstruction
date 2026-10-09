#!/bin/bash
cd "$(dirname "$0")"
pixi run marimo run notebooks/marimo_optimize_svmbir_parameters.py
