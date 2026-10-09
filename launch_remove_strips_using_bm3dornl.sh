#!/bin/bash
cd "$(dirname "$0")"
pixi run marimo run notebooks/marimo_remove_strips_using_bm3dornl.py
