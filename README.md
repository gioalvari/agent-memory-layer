# Agent Memory Layer

A transparent HTTP proxy that gives LLM agents persistent, searchable memory across sessions — entirely local, zero cloud dependencies.

Built with C++17, llama.cpp embeddings on Apple Silicon Metal, and SQLite vector storage.

## How It Works

```
Your Agent ──→ Memory Layer Proxy (port 8800) ──→ LLM Backend (port 8080)
                        │
             ┌──────────┴──────────────┐
             │ 1. Embed user query     │
             │ 2. Search past memories │
             │ 3. Inject into prompt   │
             │ 4. Forward to LLM       │
             │ 5. Capture & save       │
             └─────────────────────────┘
```

The proxy intercepts OpenAI-compatible `/v1/chat/completions` calls, searches for relevant past interactions using cosine similarity on local embeddings, injects matching context into the system prompt, and saves new conversation turns — all transparently.

**Zero code changes required in your agent** — just change `base_url`.

## Features

- 🧠 **Dual-embedding retrieval** — matches against both user questions and assistant responses
- ⏱️ **Temporal decay scoring** — recent memories rank higher, old ones gracefully fade
- 🔄 **Auto-deduplication** — near-identical memories merge instead of accumulating
- 🏷️ **Per-agent namespaces** — isolate or share memories across agents via `X-Agent-Id`
- ⚡ **Metal-accelerated embeddings** — local inference via llama.cpp, no network calls
- 📡 **Real SSE streaming** — true chunked streaming passthrough with background content capture
- 🎯 **Priority embedding queue** — search requests (HIGH) preempt background saves (NORMAL)
- 🛡️ **Input validation** — 1MB body limit, JSON schema checks, proper HTTP error codes
- 📊 **Admin API** — runtime stats, memory listing, debug injection inspection
- 🔧 **Token budget control** — configurable max tokens for injected memory context
- 📝 **Structured logging** — timestamped `[INFO/WARN/ERROR][component]` output

## Quick Start

```bash
# Clone with submodules (llama.cpp)
git clone --recursive https://github.com/youruser/agent-memory-layer.git
cd agent-memory-layer

# Build
mkdir build && cd build
cmake .. && make -j$(nproc)

# Download embedding model
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf

# Run (assumes LLM server on port 8080, e.g. llama-server)
./memory-layer --embedding-model nomic-embed-text-v1.5.Q8_0.gguf

# Your agent just changes base_url:
# client = OpenAI(base_url="http://localhost:8800/v1")
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--embedding-model` | *(required)* | Path to GGUF embedding model |
| `--backend` | `http://localhost:8080` | LLM backend URL |
| `--port` | `8800` | Proxy listen port |
| `--db` | `memories.sqlite` | SQLite database path |
| `--top-k` | `5` | Max memories to inject per request |
| `--decay-days` | `30` | Temporal decay half-life (days) |
| `--dedup-threshold` | `0.92` | Cosine similarity threshold for dedup |
| `--max-memories-per-agent` | `1000` | Eviction cap per agent namespace |
| `--max-inject-tokens` | `2048` | Token budget for injected memory block |
| `--gpu-layers` | `99` | GPU layers for embedding model |

## Agent Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="not-needed",
    default_headers={"X-Agent-Id": "my-coding-agent"}
)

response = client.chat.completions.create(
    model="any-model",
    messages=[{"role": "user", "content": "What did we discuss yesterday?"}]
)
# Proxy automatically injects relevant past context into the prompt!
```

Works with any OpenAI-compatible client: Python, TypeScript, curl, LangChain, etc.

## Admin API

Runtime introspection without stopping the server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (returns `{"status":"ok"}`) |
| `/admin/stats` | GET | Memory count, embedding queue depth, uptime |
| `/admin/memories` | GET | List memories (`?agent_id=X&limit=N&offset=M`) |
| `/admin/memories/:id` | DELETE | Remove a specific memory by ID |
| `/admin/debug/last-injection` | GET | Inspect the last memory injection payload |

```bash
# Check stats
curl http://localhost:8800/admin/stats

# List memories for a specific agent
curl "http://localhost:8800/admin/memories?agent_id=my-agent&limit=10"

# Delete a memory
curl -X DELETE http://localhost:8800/admin/memories/42
```

## How Memory Injection Looks

The proxy transparently enriches the system message:

```
[Your original system prompt]

<memory context>
Relevant past interactions (scored by relevance × recency):
- [2h ago, score=0.87] User asked: "How do I sort a vector?" → std::sort with iterators...
- [1d ago, score=0.71] User prefers PostgreSQL over MySQL for this project.
- [3d ago, score=0.54] Project uses CMake with C++17 and Catch2 for tests.
</memory context>
```

If no relevant memories are found (score < 0.3), the request passes through **unmodified**.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   HTTP Server (httplib)                    │
│  /v1/chat/completions  │  /admin/*  │  /health  │ /v1/*  │
└──────────┬─────────────┴────────────┴───────────┴────────┘
           │
     ┌─────▼──────┐     ┌──────────────────┐
     │  Injector   │────▶│  Memory Store    │
     │ (enriches   │     │  (SQLite + RAM   │
     │  prompts)   │     │   vector cache)  │
     └─────────────┘     └────────┬─────────┘
                                  │
     ┌─────────────┐     ┌───────▼──────────┐
     │ Save Queue   │────▶│ Embedding Worker │
     │ (background  │     │ (llama.cpp Metal │
     │  thread)     │     │  priority queue) │
     └─────────────┘     └──────────────────┘
```

### Components

| Component | Responsibility |
|-----------|---------------|
| **MemoryProxy** | HTTP routing, SSE streaming, input validation, request forwarding |
| **MemoryStore** | SQLite persistence, in-RAM vector cache, cosine search, eviction, dedup |
| **EmbeddingWorker** | llama.cpp model loading, priority queue (HIGH=search, NORMAL=save), thread-safe |
| **Injector** | Formats memory context block, respects token budget, inserts into messages |
| **Config** | CLI parsing, validation, sensible defaults |

### Design Decisions

- **Single-process, multi-thread**: proxy thread + embedding worker + background saver. No IPC complexity.
- **In-RAM vector index**: all embeddings loaded at startup for O(n) brute-force cosine search. Fast up to ~10k memories; no ANN index needed at this scale.
- **Priority embedding queue**: search requests (user-facing latency) preempt save requests (background). Prevents slow saves from blocking fast lookups.
- **SQLite WAL mode**: concurrent reads + single writer. Embeddings stored as BLOBs, loaded into flat float arrays on startup.
- **Graceful degradation**: if embedding model fails to load, proxy passes through all requests unmodified (no memories injected or saved).

## Technical Details

### Scoring Formula

```
cos_max   = max(cosine(query, user_emb), cosine(query, assist_emb))
age_hours = (now - created_at) / 3600.0
decay     = max(0.5, 1.0 - age_hours / (24 × decay_days))
score     = cos_max × decay
if agent matches:  score *= agent_boost (default 1.2)
filter:            score >= min_score (default 0.3)
```

### Dual-Embedding Search

Each memory stores **two** embedding vectors:
- **user_emb**: the user's question/input
- **assist_emb**: the assistant's response

Search compares query against BOTH, taking the max. This retrieves memories where either:
- The user asked something similar before (semantic question match)
- The assistant produced relevant knowledge (content match)

### Memory Namespaces

| Header | Behavior |
|--------|----------|
| `X-Agent-Id: coder` | Searches coder-specific + global memories |
| `X-Agent-Id: reviewer` | Isolated from coder's memories |
| *(no header)* | Global namespace only |

Agent-specific memories get a 1.2× scoring boost over cross-agent matches.

### Eviction & Deduplication

- When an agent exceeds `--max-memories-per-agent`, the oldest/lowest-scored memories are evicted.
- Before saving, the system checks for near-duplicate memories (cosine > `--dedup-threshold`). Duplicates are merged rather than accumulated.
- After eviction, the in-RAM vector index is recompacted to reclaim memory.

## Testing

```bash
cd build && cmake .. && make -j8

# Unit tests (25 tests)
./test/test_config
./test/test_memory_store
./test/test_cosine
./test/test_injector
./test/test_scoring

# Integration tests (12 end-to-end HTTP tests)
./test/test_integration
```

**37 total tests** covering: config parsing, memory CRUD, cosine similarity, prompt injection formatting, scoring formula, and full HTTP round-trip integration (mock backend + real proxy).

## Recommended Embedding Model

[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) — 768 dimensions, excellent quality/speed ratio on Apple Silicon.

```bash
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
```

Performance on M4 Pro: ~2ms per embedding (768-dim), ~50μs per cosine search over 1000 memories.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) — Metal acceleration
- CMake 3.20+
- C++17 compiler (Xcode Command Line Tools)
- ~500MB RAM for embedding model + vector cache

## License

MIT
