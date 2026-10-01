#!/usr/bin/env bash
# Local preview at http://localhost:8000 (PORT=8001 ./local_serve.sh for another port).
# Rebuilds on changes to content, theme and config; restart after editing the plugin.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/pelican content -s pelicanconf.py -o output --listen --autoreload --port "${PORT:-8000}"
