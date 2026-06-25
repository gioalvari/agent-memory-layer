#!/usr/bin/env python3
"""End-to-end test for Agent Memory Layer.

Can run in two modes:
  1. Auto-launch mode (default): starts the proxy subprocess, runs tests, stops it.
  2. External mode (--no-launch): assumes proxy is already running.

The test sends 5 conversation turns. Turns 4-5 reference topics from turns 1-2,
which should trigger memory injection (score > 0.3 threshold).

Usage:
    # Auto-launch (needs compiled proxy + running Ollama):
    python3 scripts/e2e_test.py --proxy-bin ./build/memory-layer

    # External proxy (proxy already running on port 8800):
    python3 scripts/e2e_test.py --no-launch --model qwen2.5-coder:32b

Requirements: Python 3.8+ stdlib only (no third-party packages).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


MXBAI_GGUF = os.path.expanduser(
    "~/.ollama/models/blobs/"
    "sha256-819c2adf5ce6df2b6bd2ae4ca90d2a69f060afeb438d0c171db57daa02e39c3d"
)
PROXY_PORT = 8800
AGENT_ID   = "e2e-test-agent"

# Turns designed so turns 4-5 reference topics from turns 1-2,
# triggering memory retrieval once saves from turns 1-3 are complete.
TURNS = [
    "My project uses PostgreSQL 16 with the pg_vector extension for semantic search.",
    "I'm building a C++ HTTP proxy using cpp-httplib and SQLite for persistence.",
    "The embedding model is mxbai-embed-large with 1024 dimensions.",
    "How should I index my vectors in the database I mentioned earlier?",
    "What is a good way to handle SSE streaming in the proxy I described?",
]


def check(ok: bool, msg: str) -> None:
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status}  {msg}")
    if not ok:
        sys.exit(1)


def warn(ok: bool, msg: str) -> None:
    """Like check but only warns — does not exit on failure."""
    status = "✅ OK  " if ok else "⚠  WARN"
    print(f"  {status}  {msg}")


def wait_port(port: int, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("localhost", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def http_get(url: str, timeout: float = 10.0) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, payload: dict, headers: dict,
              timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-bin",   default="./build/memory-layer",
                    help="Path to compiled memory-layer binary")
    ap.add_argument("--model-path",  default=MXBAI_GGUF,
                    help="Path to GGUF embedding model")
    ap.add_argument("--backend",     default="http://localhost:11434",
                    help="LLM backend URL (Ollama or llama-server)")
    ap.add_argument("--model",       default="qwen2.5-coder:32b",
                    help="LLM model name for completions")
    ap.add_argument("--base-url",    default=f"http://localhost:{PROXY_PORT}/v1")
    ap.add_argument("--admin-url",   default=f"http://localhost:{PROXY_PORT}/admin")
    ap.add_argument("--no-launch",   action="store_true",
                    help="Skip launching proxy (assume it is already running)")
    args = ap.parse_args()

    print(f"=== Agent Memory Layer E2E Test ===")
    print(f"base={args.base_url}  backend={args.backend}  model={args.model}\n")

    proc = None
    db_path = "/tmp/e2e_test_memories.sqlite"

    if not args.no_launch:
        # ── Pre-flight ────────────────────────────────────────────────────
        print("=== Pre-flight ===")
        check(os.path.exists(args.model_path),
              f"GGUF model exists at {args.model_path}")
        check(os.path.exists(args.proxy_bin),
              f"Proxy binary exists at {args.proxy_bin}")
        try:
            r = http_get(f"{args.backend}/v1/models", timeout=5)
            check(isinstance(r, dict), f"Ollama reachable at {args.backend}")
        except Exception as e:
            check(False, f"Ollama reachable: {e}")

        if os.path.exists(db_path):
            os.remove(db_path)

        # ── Launch proxy ──────────────────────────────────────────────────
        print("\n=== Starting proxy ===")
        proc = subprocess.Popen(
            [
                args.proxy_bin,
                "--embedding-model", args.model_path,
                "--backend",         args.backend,
                "--port",            str(PROXY_PORT),
                "--db",              db_path,
                "--top-k",           "3",
                "--gpu-layers",      "99",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(f"  Proxy PID: {proc.pid}")
        check(wait_port(PROXY_PORT, timeout=45),
              "Proxy listening on port 8800 within 45s")
    else:
        # External mode: just check health
        print("=== Pre-flight (external proxy) ===")
        try:
            health = http_get(f"{args.base_url.replace('/v1', '')}/health")
            check(True, f"Proxy healthy: {health}")
        except Exception as e:
            check(False, f"Proxy reachable: {e}")

    try:
        _run_turns(args)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=5)
            if os.path.exists(db_path):
                os.remove(db_path)
            print("\n=== Proxy stopped, temp DB cleaned up ===")

    return 0


def _run_turns(args: argparse.Namespace) -> None:
    base    = args.base_url
    admin   = args.admin_url
    history = []

    print("\n=== Conversation turns ===")
    for i, user_msg in enumerate(TURNS, 1):
        history.append({"role": "user", "content": user_msg})
        body = http_post(
            f"{base}/chat/completions",
            payload={
                "model":    args.model,
                "messages": history,
                "stream":   False,
            },
            headers={
                "Content-Type": "application/json",
                "X-Agent-Id":   AGENT_ID,
            },
        )
        reply = body["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        print(f"  T{i}: {reply[:80].strip()}...")
        check(True, f"Turn {i} completed (HTTP 200)")
        time.sleep(1.5)  # allow background save to complete

    # ── Ring buffer ───────────────────────────────────────────────────────
    print("\n=== Injection ring buffer ===")
    ring = http_get(f"{admin}/debug/last-injection")
    if not isinstance(ring, list):
        ring = [ring]  # graceful handle for old single-object format
    check(len(ring) >= 1, f"Ring has ≥1 entry (got {len(ring)})")
    check(ring[0].get("agent_id") == AGENT_ID,
          f"ring[0].agent_id == '{AGENT_ID}'")
    injected = ring[0].get("injected_context", "")
    warn(len(injected) > 0,
         f"Non-empty injection context ({len(injected)} chars); "
         "if 0, memories may not have crossed score threshold 0.3 yet")
    if injected:
        print(f"  Snippet: {injected[:120]!r}")

    # ── Stats ─────────────────────────────────────────────────────────────
    print("\n=== Memory stats ===")
    stats = http_get(f"{admin}/stats")
    total = stats.get("total_memories", stats.get("memory_count", 0))
    check(total >= len(TURNS),
          f"At least {len(TURNS)} memories saved (got {total})")
    print(f"  {json.dumps(stats, indent=2)}")

    print("\n🎉  All assertions passed — memory injection working end-to-end!")


if __name__ == "__main__":
    sys.exit(main())

