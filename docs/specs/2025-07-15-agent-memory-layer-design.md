# Agent Memory Layer — Design Spec

## Overview

A transparent HTTP proxy that gives LLM agents persistent memory across sessions.
Sits between agent code and any OpenAI-compatible LLM server, intercepting requests to
inject relevant past context and saving new interactions for future retrieval.

**Key differentiators vs Mem0/Zep:**
- Fully local (no cloud, no API keys)
- Zero SDK integration — just change `base_url`
- Dual-embedding retrieval (query similarity + knowledge similarity)
- Temporal decay ranking
- C++ single-binary, Metal-accelerated embeddings

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Agent Memory Layer                      │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │  HTTP    │───▶│  Memory   │───▶│   LLM Backend    │  │
│  │  Proxy   │◀───│  Engine   │◀───│   (forward)      │  │
│  │(httplib) │    │           │    │                   │  │
│  └──────────┘    └─────┬─────┘    └──────────────────┘  │
│                        │                                  │
│              ┌─────────┴─────────┐                       │
│              │                   │                        │
│         ┌────▼────┐      ┌──────▼──────┐                │
│         │ SQLite  │      │  Embedding  │                │
│         │  Store  │      │   Worker    │                │
│         │         │      │ (llama.cpp) │                │
│         └─────────┘      └─────────────┘                │
└─────────────────────────────────────────────────────────┘
```

### Threading Model (Approach C: Monolith with Thread Pool)

- **Main thread**: httplib server, accepts connections
- **Request handler threads**: httplib's built-in thread pool (one per concurrent request)
- **Embedding worker thread**: dedicated thread with llama.cpp context, processes embed jobs from a queue
- **Background saver thread**: post-response memory saving (non-blocking)

## Request Flow

1. Agent sends `POST /v1/chat/completions` to proxy (port 8800)
2. Proxy extracts last user message + `X-Agent-Id` header
3. Memory Engine embeds the query via embedding worker thread
4. Searches SQLite for relevant memories (dual-embedding: user_emb + assist_emb)
5. Injects top-K memories into system message (if score > 0.3 threshold)
6. Forwards modified request to backend LLM
7. Receives response (streaming or batch), forwards immediately to agent
8. Post-response: saves the full turn (user + assistant) with embeddings in background

## Data Model

```sql
CREATE TABLE memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id     TEXT,              -- NULL = global memory
    created_at   REAL NOT NULL,     -- unix timestamp
    updated_at   REAL NOT NULL,     -- for dedup/merge tracking
    user_text    TEXT NOT NULL,     -- original user message
    assist_text  TEXT NOT NULL,     -- assistant response
    user_emb     BLOB NOT NULL,     -- float32[768] embedding of user_text
    assist_emb   BLOB NOT NULL,     -- float32[768] embedding of assist_text
    access_count INTEGER DEFAULT 1  -- popularity counter for retrieval boost
);

CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_created ON memories(created_at DESC);
```

### Embedding Model

- Model: `nomic-embed-text-v1.5.Q8_0.gguf` (137M params, 768 dimensions)
- Vector size: 768 × 4 bytes = 3072 bytes per embedding
- Search: brute-force cosine similarity (< 1ms for 10k memories)

### Ranking Formula

```
score = max(cosine(query_emb, user_emb), cosine(query_emb, assist_emb)) × decay(age)
decay(hours) = max(0.5, 1.0 - hours / (24 × decay_days))
```

The `max()` over both embeddings implements dual-retrieval: finds memories where either
the user asked something similar OR the assistant produced relevant knowledge.

### Deduplication

Before inserting a new memory:
1. Embed the new turn
2. Search for existing memory with `max_cosine > dedup_threshold` (default 0.92)
3. If found: update `updated_at`, increment `access_count` (no new row)
4. If not found: insert new row

### Limits

- Max memories per agent: 1000 (configurable, FIFO eviction after limit)
- Max global memories: 5000
- Top-K retrieval per request: 5 (configurable)

## Injection Format

Memories are prepended to the system message:

```
[Original system prompt]

<memory context>
Relevant past interactions:
- [2h ago] User asked about database indexing. Response: Explained B-trees vs hash indexes with examples.
- [1d ago] User prefers PostgreSQL over MySQL for this project.
- [3d ago] The project uses Python 3.12 with FastAPI.
</memory context>
```

### Injection Rules

1. No existing system message → create one with only memory context
2. Existing system message → append memory context after `\n\n`
3. No relevant memories (all scores < 0.3) → forward request unmodified (zero overhead)
4. Each memory truncated to max 200 chars user + 200 chars assistant
5. Estimated token budget: ~500 tokens for 5 memories (negligible in 32k context)

## Streaming Support

```
Agent → Proxy → Backend (stream request)
                         ↓
              [SSE chunks: data: {...}]
                         ↓
Proxy buffer ←── accumulate tokens ──── immediate forward to agent
                         ↓
              [data: [DONE]]
                         ↓
         Background: save user+assistant, embed, store
```

- **Zero added latency**: every SSE chunk is forwarded immediately
- **Buffer accumulation**: content deltas collected in parallel for post-save
- **Async save**: embedding + SQLite insert happens after response completes
- **Non-streaming**: simpler — extract from full JSON response, forward, save

## Error Handling

- Backend unreachable → 502 to agent, no memory saved
- Backend returns 4xx/5xx → forward error to agent, no memory saved
- Embedding model fails → log error, forward request without memory injection
- SQLite write fails → log error, don't crash (memory is best-effort)

## CLI Interface

```bash
./memory-layer \
  --backend http://localhost:8080 \
  --port 8800 \
  --embedding-model nomic-embed.gguf \
  --db memories.sqlite \
  --top-k 5 \
  --decay-days 30 \
  --dedup-threshold 0.92 \
  --max-memories-per-agent 1000 \
  --gpu-layers 99
```

## Project Structure

```
memory-layer/
├── CMakeLists.txt
├── README.md
├── include/
│   └── memorylayer/
│       ├── config.h          # struct Config + CLI parsing
│       ├── memory_store.h    # SQLite wrapper + search logic
│       ├── embedding.h       # llama.cpp embedding worker
│       ├── proxy.h           # HTTP proxy + streaming
│       └── injector.h        # memory formatting + injection
├── src/
│   ├── main.cpp
│   ├── config.cpp
│   ├── memory_store.cpp
│   ├── embedding.cpp
│   ├── proxy.cpp
│   └── injector.cpp
├── deps/                     # llama.cpp, httplib, sqlite3, nlohmann/json
└── test/
    └── test_memory_store.cpp
```

## Dependencies (Vendored)

| Dependency | Purpose | Integration |
|------------|---------|-------------|
| llama.cpp | Embedding inference via Metal | git submodule |
| cpp-httplib | HTTP server + HTTP client | header-only |
| sqlite3 | Memory storage | single .c/.h file |
| nlohmann/json | JSON parsing | header-only |

## Agent Usage

```python
# Before (direct to LLM):
client = OpenAI(base_url="http://localhost:8080/v1")

# After (through memory layer):
client = OpenAI(
    base_url="http://localhost:8800/v1",
    default_headers={"X-Agent-Id": "coder-agent"}
)
# No other code changes needed!
```

## Memory Namespaces

- `X-Agent-Id: coder` → searches agent-specific memories first, then global
- `X-Agent-Id: reviewer` → isolated from coder's memories unless global
- No header → treated as global namespace only
- Priority: agent-specific memories get 1.2x score boost over global matches

## Non-Goals (YAGNI)

- No web UI for memory management
- No CLI for querying memories
- No export/import functionality
- No multi-model support (one embedding model)
- No authentication on the proxy
- No conversation threading (each request is independent)

## Success Criteria

1. An agent using the proxy gets contextually relevant past info injected automatically
2. Zero code changes required in agent code (just `base_url` swap)
3. < 50ms added latency on request path (embedding + search)
4. Streaming responses have zero added latency (inject only on request, not response)
5. Works with any OpenAI-compatible backend (llama.cpp server, Ollama, vLLM, etc.)
