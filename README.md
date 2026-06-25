# Agent Memory Layer

A transparent HTTP proxy that gives LLM agents persistent memory across sessions.

## How It Works

```
Your Agent → Memory Layer Proxy (port 8800) → LLM Server (port 8080)
                    │
         ┌──────────┴──────────┐
         │ 1. Embed user query │
         │ 2. Search memories  │
         │ 3. Inject context   │
         │ 4. Forward request  │
         │ 5. Save response    │
         └─────────────────────┘
```

The proxy intercepts OpenAI-compatible API calls, searches for relevant past interactions,
injects them into the system prompt, and saves new interactions — all transparently.

**Zero code changes required** — just change your `base_url`.

## Features

- 🧠 **Dual-embedding retrieval** — finds similar past questions AND relevant knowledge
- ⏱️ **Temporal decay** — recent memories rank higher
- 🔄 **Auto-deduplication** — similar memories merge instead of accumulating
- 🏷️ **Per-agent namespaces** — isolate or share memories between agents
- ⚡ **Metal-accelerated** — local embeddings via llama.cpp on Apple Silicon
- 📡 **Streaming support** — SSE forwarding with background content capture

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. && make -j$(nproc)

# Run (requires an embedding model + LLM server on port 8080)
./memory-layer --embedding-model path/to/nomic-embed-text-v1.5.Q8_0.gguf

# Your agent just changes base_url:
# client = OpenAI(base_url="http://localhost:8800/v1")
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--embedding-model` | (required) | Path to GGUF embedding model |
| `--backend` | `http://localhost:8080` | LLM backend URL |
| `--port` | `8800` | Proxy listen port |
| `--db` | `memories.sqlite` | SQLite database path |
| `--top-k` | `5` | Memories to inject per request |
| `--decay-days` | `30` | Temporal decay half-life |
| `--dedup-threshold` | `0.92` | Cosine threshold for deduplication |
| `--max-memories-per-agent` | `1000` | Max memories per agent |
| `--gpu-layers` | `99` | GPU layers for embedding model |

## Agent Integration

```python
from openai import OpenAI

# Just change the base_url and add an agent ID header:
client = OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="not-needed",
    default_headers={"X-Agent-Id": "my-agent"}
)

response = client.chat.completions.create(
    model="any-model",
    messages=[{"role": "user", "content": "What did we discuss yesterday?"}]
)
# The proxy automatically injects relevant past context!
```

## How Memory Injection Looks

The proxy transparently modifies the system message:

```
[Your original system prompt]

<memory context>
Relevant past interactions:
- [2h ago] How do I sort a vector? Response: Use std::sort with iterators...
- [1d ago] User prefers PostgreSQL over MySQL for this project.
</memory context>
```

If no relevant memories are found (score < 0.3), the request passes through unmodified.

## Architecture

- **C++17** single binary
- **llama.cpp** for local embedding inference (Metal/GPU)
- **SQLite** for persistent memory storage (WAL mode)
- **cpp-httplib** for HTTP server + client
- **Dedicated threads**: embedding worker + background saver (non-blocking proxy)

## Recommended Embedding Model

[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) — 768 dimensions, excellent quality/speed on Apple Silicon.

```bash
# Download
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
```

## Technical Details

### Dual-Embedding Search

Each memory stores two embeddings:
- **user_emb**: vector of the user's question
- **assist_emb**: vector of the assistant's response

When searching, the query is compared against BOTH. This finds memories where:
- The user asked something similar before (query ↔ user_emb)
- The assistant produced relevant knowledge (query ↔ assist_emb)

### Ranking Formula

```
score = max(cosine(query, user_emb), cosine(query, assist_emb)) × decay(age)
decay(hours) = max(0.5, 1.0 - hours / (24 × decay_days))
```

Agent-specific memories get a 1.2× score boost over global matches.

### Memory Namespaces

- `X-Agent-Id: coder` → agent-specific + global memories
- `X-Agent-Id: reviewer` → isolated from coder unless global
- No header → global namespace only
