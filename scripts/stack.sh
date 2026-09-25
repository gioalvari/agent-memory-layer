#!/usr/bin/env bash
# Run memory-layer in front of an LLM backend as one process tree.
#
# Usage:
#   scripts/stack.sh <port> <inject-mode> <embedding-model.gguf> -- <backend cmd...>
#
# The backend command may contain {bport}, replaced by a free local port. The
# proxy listens on <port>, uses a fresh temporary SQLite database, and is only
# started once the backend answers GET /health. SIGTERM/SIGINT stop both.
# Intended for benchmarks (e.g. local-llm-bench `prefix-sharing` targets).
set -euo pipefail

if [ $# -lt 5 ] || [ "$4" != "--" ]; then
	echo "usage: $0 <port> <inject-mode> <embedding-model> -- <backend cmd...>" >&2
	exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEMORY_LAYER="${MEMORY_LAYER_BIN:-$SCRIPT_DIR/../build/memory-layer}"
PORT="$1"
MODE="$2"
EMBED="$3"
shift 4

BPORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
BACKEND=()
for arg in "$@"; do BACKEND+=("${arg//\{bport\}/$BPORT}"); done

DB_DIR="$(mktemp -d)"
BACKEND_PID=""
PROXY_PID=""

cleanup() {
	[ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true
	[ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
	wait 2>/dev/null || true
	rm -rf "$DB_DIR"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

echo "[stack] backend on port $BPORT: ${BACKEND[*]}"
"${BACKEND[@]}" &
BACKEND_PID=$!

for _ in $(seq 1 600); do
	if curl -sf "http://127.0.0.1:$BPORT/health" >/dev/null 2>&1; then break; fi
	if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
		echo "[stack] backend exited during startup" >&2
		exit 1
	fi
	sleep 0.2
done

echo "[stack] memory-layer on port $PORT (inject-mode=$MODE, db=$DB_DIR)"
"$MEMORY_LAYER" \
	--embedding-model "$EMBED" \
	--backend "http://127.0.0.1:$BPORT" \
	--port "$PORT" \
	--db "$DB_DIR/memories.sqlite" \
	--inject-mode "$MODE" &
PROXY_PID=$!

wait "$PROXY_PID"
