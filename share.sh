#!/usr/bin/env bash
# Share the local preview publicly through ngrok (PORT=8001 ./share.sh for another port).
# Starts ./serve.sh unless something is already serving on the port; Ctrl-C stops both.
# Anyone with the ngrok link can see the site while this runs.
set -euo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-8000}

if ! curl -s -o /dev/null "http://localhost:$PORT/"; then
  # own process group, so Ctrl-C/exit stops pelican's autoreload children too
  PORT=$PORT setsid ./serve.sh > /dev/null 2>&1 &
  server=$!
  trap 'kill -- -$server 2>/dev/null' EXIT
  for _ in $(seq 40); do
    curl -s -o /dev/null "http://localhost:$PORT/" && break
    kill -0 $server 2>/dev/null || { echo "preview failed to start; run ./serve.sh to see why" >&2; exit 1; }
    sleep 0.25
  done
fi

ngrok http "$PORT"
