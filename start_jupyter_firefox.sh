#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec pixi run lab --ServerApp.browser=firefox "$@"
