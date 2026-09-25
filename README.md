# Agent Memory Layer

[![CI](https://github.com/gioalvari/agent-memory-layer/actions/workflows/ci.yml/badge.svg)](https://github.com/gioalvari/agent-memory-layer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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
- ⚡ **Batch embedding** — up to 8 texts encoded in a single `llama_decode`; ~3× throughput at 8 concurrent texts
- 🔍 **Injection history ring** — last 10 memory injections inspectable via admin API (newest first)
- 🛡️ **Input validation** — 1MB body limit, JSON schema checks, proper HTTP error codes
- 📊 **Admin API** — runtime stats, memory listing, debug injection inspection
- 🔧 **Token budget control** — configurable max tokens for injected memory context
- 📝 **Structured logging** — timestamped `[INFO/WARN/ERROR][component]` output

## Quick Start

```bash
# Clone with submodules (llama.cpp)
git clone --recursive https://github.com/gioalvari/agent-memory-layer.git
cd agent-memory-layer

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# Download embedding model
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf

# Run (assumes LLM server on port 8080, e.g. llama-server)
./build/memory-layer --embedding-model nomic-embed-text-v1.5.Q8_0.gguf

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
| `--inject-mode` | `system` | `system`: append memories to the system prompt. `suffix`: prepend them to the last user message, keeping the system prompt and history byte-identical for backend prefix caching |

## Running with Ollama

The easiest real-world setup uses [Ollama](https://ollama.ai) as the LLM backend:

```bash
# Pull models (if not already done)
ollama pull qwen2.5-coder:32b      # or any compatible model
ollama pull mxbai-embed-large      # embedding model

# The mxbai GGUF is already on disk after pull — find it:
GGUF=$(ls ~/.ollama/models/blobs/ | while read f; do
    xxd -l4 ~/.ollama/models/blobs/$f 2>/dev/null | grep -q "GGUF" \
    && echo ~/.ollama/models/blobs/$f; done | head -1)

# Or download nomic-embed-text directly (recommended, ~274MB):
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf

# Start Ollama (if not running)
ollama serve &

# Start memory-layer
./build/memory-layer \
    --embedding-model nomic-embed-text-v1.5.Q4_K_M.gguf \
    --backend http://localhost:11434 \
    --port 8800

# Smoke test
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Agent-Id: test-agent" \
  -d '{"model":"qwen2.5-coder:32b","messages":[{"role":"user","content":"My project uses CMake and C++17."}]}'
```

### VS Code / Continue Integration

Add to `.continuerc.json`:

```json
{
  "models": [{
    "title": "Memory-Aware Qwen",
    "provider": "openai",
    "model": "qwen2.5-coder:32b",
    "apiBase": "http://localhost:8800/v1",
    "apiKey": "not-needed",
    "requestOptions": {
      "headers": { "X-Agent-Id": "vscode-agent" }
    }
  }]
}
```

### Cursor Integration

Settings → Models → Add Custom Model:
- **API Base**: `http://localhost:8800/v1`
- **Model**: `qwen2.5-coder:32b`
- **API Key**: `not-needed`

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
| `/admin/agents` | GET | Agent namespaces with memory counts |
| `/admin/ui` | GET | Embedded web dashboard |
| `/admin/memories` | GET | List memories (`?agent_id=X&limit=N&offset=M`) |
| `/admin/memories/:id` | DELETE | Remove a specific memory by ID |
| `/admin/debug/last-injection` | GET | JSON array of last 10 injections, newest first |

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

With `--inject-mode suffix` the same block is instead prepended to the **last user
message**, so everything before it (system prompt and history) is identical to the
previous request and stays in the backend's prefix/KV cache.

### Benchmark: injection position

8 agents with a shared ~1,500-token system prompt, 4 turns each, proxy in front of
[RadixForge](https://github.com/gioalvari/radixforge), Qwen2.5-0.5B on an M4 Pro
(median time to first token on follow-up turns, 1,440 requests, 0 failures):

| Concurrency | No proxy | `--inject-mode system` | `--inject-mode suffix` |
|---:|---:|---:|---:|
| 1 | 21.6 ms | 65.5 ms | **59.6 ms** |
| 8 | 110.2 ms | 317.4 ms | **227.5 ms** |

Suffix mode cuts prefilled tokens by 22–25% on later turns. The remaining overhead
is the memory block itself, which changes on every request. Method and full results:
[local-llm-bench](https://github.com/gioalvari/local-llm-bench)
(`results/qwen-0.5b-q4-memory-injection-m4-pro.md`).

Run the whole stack (proxy + backend) as one process with `scripts/stack.sh`:

```bash
./scripts/stack.sh 8800 suffix nomic-embed-text-v1.5.Q8_0.gguf -- \
  llama-server -m model.gguf --port {bport}
```

## Architecture

### Full Request Lifecycle

```
 ┌──────────────────────────────────────────────────────────────────┐
 │               Agent / VS Code / Cursor / curl                     │
 │   POST /v1/chat/completions  {"messages":[...], "model":"..."}    │
 └────────────────────────────┬─────────────────────────────────────┘
                              │ HTTP (port 8800)
 ┌────────────────────────────▼─────────────────────────────────────┐
 │              MemoryProxy  ── main thread                          │
 │  1. Validate (size ≤ 1MB, JSON schema, required fields)          │
 │  2. Extract user_text + agent_id (X-Agent-Id header)             │
 │  3. embed(user_text, HIGH) ─────────────────────────────────┐    │
 └────────────────────────────┬────────────────────────────────┼────┘
              wait future.get()│                                │
 ┌────────────────────────────▼────────────────────────────────┼────┐
 │           EmbeddingWorker  ── background thread             │    │
 │  ┌──────────────────────────────────────────────────────┐   │    │
 │  │  Priority Queue                                       │   │    │
 │  │  ■ HIGH   (search queries — drain first, low latency)│   │    │
 │  │  □ NORMAL (background saves — drain after)           │   │    │
 │  └───────────────────────────┬──────────────────────────┘   │    │
 │            ┌─────────────────▼──────────────────────────┐   │    │
 │            │ Batch path  (≤8 texts, total ≤2048 tokens): │   │    │
 │            │   1 × llama_decode  →  N embeddings (fast)  │   │    │
 │            │ Serial fallback     (token budget overflow): │   │    │
 │            │   N × llama_decode  (1 text each)            │   │    │
 │            └──────────────────────────────────────────────┘   │    │
 └─────────────────────────────────────────────────────────────┼────┘
                              query_emb ──────────────────────-┘
 ┌──────────────────────────────────────────────────────────────────┐
 │              MemoryStore  ── shared (mutex-guarded)               │
 │  4. L2-normalize query_emb                                        │
 │  5. Scan in-RAM float cache:                                      │
 │       score = max(dot(q, user_emb), dot(q, assist_emb))          │
 │             × decay(age_hours, decay_days)                        │
 │             × agent_boost (1.2× if agent_id matches)             │
 │  6. Return top-K where score ≥ min_score (default 0.3)           │
 └────────────────────────────┬─────────────────────────────────────┘
                              │ Vec<SearchResult>
 ┌────────────────────────────▼─────────────────────────────────────┐
 │                         Injector                                  │
 │  7. Format: "- [Xh ago, score=Y] User: ... → Asst: ..."          │
 │  8. Trim to --max-inject-tokens budget                            │
 │  9. Prepend block to messages[0] (system prompt)                  │
 └────────────────────────────┬─────────────────────────────────────┘
                              │ enriched request JSON
 ┌────────────────────────────▼─────────────────────────────────────┐
 │         LLM Backend  (Ollama / llama-server, port 8080+)          │
 │  10. Forward enriched JSON                                        │
 │  11. Stream SSE chunks → proxy → agent (real token-by-token)     │
 │  12. Background: capture assistant text → embed(NORMAL) → SQLite │
 └──────────────────────────────────────────────────────────────────┘
```

### Batch Embedding Worker

```
Queue HIGH   [search-q]
Queue NORMAL [save-u1] [save-a1] [save-u2] [save-a2] …

Worker drains ≤8 jobs per iteration (HIGH fully before NORMAL):

  ┌── job1  job2  job3  job4 ──────────────────────────┐
  │     │     │     │     │                             │
  │     └─────┴─────┴─────┘                             │
  │         one llama_decode()  (1 Metal dispatch)      │
  │         ↓    ↓    ↓    ↓                            │
  │       emb1 emb2 emb3 emb4   (L2-normalized)        │
  └─────────────────────────────────────────────────────┘
  If total_tokens > 2048 → graceful serial fallback.
```

### Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **MemoryProxy** | `src/proxy.cpp` | HTTP routing, SSE passthrough, validation, injection ring buffer (10 entries) |
| **MemoryStore** | `src/memory_store.cpp` | SQLite WAL, flat float cache, dot-product search, LRU eviction, dedup |
| **EmbeddingWorker** | `src/embedding.cpp` | llama.cpp loader, priority queue (HIGH=search, NORMAL=save), batch decode (≤8/call) |
| **Injector** | `src/injector.cpp` | Memory block formatting, token budget enforcement |
| **Config** | `src/config.cpp` | CLI parsing, validation, defaults |

### Design Decisions

- **Single-process, multi-thread**: proxy thread + embedding worker + background saver. No IPC complexity.
- **In-RAM vector index**: flat `float[]` array, O(n) dot-product scan (unit vectors → cosine = dot). Fast up to ~10k memories; no ANN index needed at this scale.
- **Priority embedding queue**: search requests (HIGH) drain fully before save requests (NORMAL). Prevents slow saves from blocking fast lookups.
- **Batch decode**: ≤8 texts encoded in a single `llama_decode` call when total tokens ≤ 2048 — ~3× throughput at N=8 on an M4 Pro. Graceful serial fallback on overflow.
- **SQLite WAL mode**: concurrent reads + single writer. Embeddings stored as BLOBs, loaded into flat float arrays on startup.
- **Injection ring buffer**: last 10 injection events stored as newest-first deque, inspectable at `/admin/debug/last-injection` without log parsing.
- **Graceful degradation**: if embedding model fails to load, proxy passes through all requests unmodified.

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
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

Embedding benchmarks (need a real GGUF model, not run by `ctest`):

```bash
./build/bench_embedding <model.gguf>             # serial vs concurrent, N = 1..8
./build/test/bench_embedding_batch <model.gguf>  # serial vs batch over n_texts
```

**37 total tests** covering: config parsing, memory CRUD, cosine similarity, prompt injection formatting, scoring formula, and full HTTP round-trip integration (mock backend + real proxy).

## Recommended Embedding Model

[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) — 768 dimensions, excellent quality/speed ratio on Apple Silicon.

```bash
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
```

Performance on M4 Pro with `nomic-embed-text` (768-dim, F16 GGUF from Ollama), measured with
`./build/bench_embedding <model.gguf>` (batch output checked against serial: cosine = 1.000):

| N texts | Serial ms/text | Batch ms/text | Speedup |
|---------|---------------|--------------|---------|
| 1       | 3.97          | 3.90         | 1.0×    |
| 2       | 3.95          | 2.76         | 1.4×    |
| 4       | 3.92          | 2.01         | 2.0×    |
| 8       | 3.93          | 1.30         | **3.0×** |

Single-text latency is ~4 ms. Per-memory cosine search: ~50μs over 1000 memories.

## Known Limitations

- **macOS / Apple Silicon only**: Metal is forced on in CMake; Linux/CUDA builds are untested.
- **OpenAI Chat Completions only**: memory is applied to `/v1/chat/completions`; other `/v1/*` paths (including Anthropic-style `/v1/messages`) are forwarded unmodified.
- **Brute-force search**: O(n) scan over an in-RAM float cache; fine up to ~10k memories per process, no ANN index.
- **Raw turns, not facts**: memories are stored as user/assistant turns; there is no LLM-based fact extraction or contradiction handling.
- **Injected block is never cached**: retrieved memories change on every request, so the backend always recomputes them. With the default `--inject-mode system` the conversation history after the system prompt is recomputed as well; `--inject-mode suffix` avoids that (see benchmark below).
- **No encryption at rest**: the SQLite database stores memories in plain text.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) — Metal acceleration
- CMake 3.20+
- C++17 compiler (Xcode Command Line Tools)
- ~500MB RAM for embedding model + vector cache

## License

MIT
