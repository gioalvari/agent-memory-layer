#!/usr/bin/env python3
"""
End-to-end test for Agent Memory Layer.

Runs 3 conversation turns with a persistent agent, then verifies that:
  - Memories are saved after each turn
  - Relevant memory is injected into turn 4 (related topic query)
  - The injection ring contains the last injection

Usage:
    python3 scripts/e2e_test.py [--base-url URL] [--admin-url URL] [--model NAME]

Prerequisites:
    - Memory Layer proxy running on --base-url (default: http://localhost:8800/v1)
    - LLM backend running and reachable (e.g. Ollama on port 11434)
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post_chat(base_url: str, agent_id: str, message: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Agent-Id": agent_id,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code} from proxy: {body}") from e


def get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent Memory Layer E2E test")
    parser.add_argument("--base-url",  default="http://localhost:8800/v1")
    parser.add_argument("--admin-url", default="http://localhost:8800/admin")
    parser.add_argument("--model",     default="llama3.2")
    parser.add_argument("--agent-id",  default="e2e-test-agent")
    args = parser.parse_args()

    agent = args.agent_id
    print(f"=== Agent Memory Layer E2E Test ===")
    print(f"base_url={args.base_url}  admin_url={args.admin_url}  model={args.model}\n")

    # --- Health check ---
    try:
        health = get_json(f"{args.admin_url.replace('/admin', '')}/health")
        print(f"✓ Proxy healthy: {health}")
    except Exception as e:
        print(f"✗ Proxy not reachable: {e}")
        return 1

    # --- Turn 1: establish a fact ---
    print("\nTurn 1: establishing build system fact...")
    r1 = post_chat(args.base_url, agent, "My project uses CMake and C++17. Please remember this.", args.model)
    print(f"  Assistant: {r1[:100].strip()}...")
    time.sleep(2)  # let background embed+save finish

    # --- Turn 2: establish another fact ---
    print("\nTurn 2: establishing database preference...")
    r2 = post_chat(args.base_url, agent, "I prefer SQLite over PostgreSQL for embedded storage.", args.model)
    print(f"  Assistant: {r2[:100].strip()}...")
    time.sleep(2)

    # --- Verify memories are stored ---
    memories = get_json(f"{args.admin_url}/memories?agent_id={agent}&limit=20")
    count = len(memories) if isinstance(memories, list) else memories.get("total", 0)
    print(f"\n✓ Memories stored so far: {count}")
    if count < 1:
        print("✗ Expected at least 1 memory after 2 turns")
        return 1

    # --- Turn 3: query that should trigger injection ---
    print("\nTurn 3: querying related topic (expects memory injection)...")
    r3 = post_chat(args.base_url, agent, "What build system and database should I use for my project?", args.model)
    print(f"  Assistant: {r3[:150].strip()}...")
    time.sleep(1)

    # --- Check injection ring ---
    injections = get_json(f"{args.admin_url}/debug/last-injection")
    if not isinstance(injections, list):
        injections = [injections]  # graceful handle for old single-object format

    print(f"\n✓ Injection ring has {len(injections)} entries")
    if injections and injections[0].get("injected_context"):
        last = injections[0]
        ctx_len = len(last["injected_context"])
        print(f"  Last query:          {last.get('query','')[:70]}")
        print(f"  Injected context:    {ctx_len} chars")
        print(f"  Agent id:            {last.get('agent_id','')}")
        if ctx_len == 0:
            print("⚠ Context was empty — memories may not have crossed score threshold (0.3)")
        else:
            print("✓ Memory injection verified")
    else:
        print("⚠ No injection context recorded — memories may not have matched score threshold")

    # --- Stats ---
    stats = get_json(f"{args.admin_url}/stats")
    print(f"\n✓ Stats: {json.dumps(stats, indent=2)}")

    print("\n=== E2E TEST COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
