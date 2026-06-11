#!/bin/bash
cd "$(dirname "$0")"
pixi run marimo run notebooks/optimize_svmbir_parameters.py
