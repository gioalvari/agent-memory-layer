# Agent Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a transparent HTTP proxy in C++ that gives LLM agents persistent memory by intercepting OpenAI API calls, injecting relevant past context, and saving new interactions.

**Architecture:** Single C++ binary with httplib (HTTP proxy + client), llama.cpp (embedding via Metal), SQLite (storage). Monolith with thread pool — embedding worker on dedicated thread, httplib handles concurrent requests, background thread for async saves.

**Tech Stack:** C++17, CMake, llama.cpp (submodule), cpp-httplib (header-only), sqlite3 (vendored), nlohmann/json (header-only)

---

## File Structure

```
memory-layer/
├── CMakeLists.txt                    # Top-level build config
├── README.md                         # Usage documentation
├── include/
│   └── memorylayer/
│       ├── config.h                  # Config struct + parse_args()
│       ├── memory_store.h            # MemoryStore class (SQLite CRUD + search)
│       ├── embedding.h               # EmbeddingWorker class (llama.cpp thread)
│       ├── injector.h                # format_memory_context() + inject_into_messages()
│       └── proxy.h                   # MemoryProxy class (HTTP proxy + streaming)
├── src/
│   ├── main.cpp                      # Entry point, wires components
│   ├── config.cpp                    # CLI parsing implementation
│   ├── memory_store.cpp              # SQLite operations + cosine search
│   ├── embedding.cpp                 # Embedding worker thread loop
│   ├── injector.cpp                  # Memory formatting logic
│   └── proxy.cpp                     # HTTP proxy, streaming, response capture
├── deps/
│   ├── llama.cpp/                    # git submodule
│   ├── httplib.h                     # cpp-httplib header
│   ├── sqlite3.c                     # SQLite amalgamation
│   ├── sqlite3.h                     # SQLite header
│   └── json.hpp                      # nlohmann/json header
└── test/
    ├── CMakeLists.txt                # Test build config
    ├── test_config.cpp               # Config parsing tests
    ├── test_memory_store.cpp         # Store CRUD + search tests
    ├── test_injector.cpp             # Injection formatting tests
    └── test_cosine.cpp               # Cosine similarity math tests
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `CMakeLists.txt`
- Create: `deps/` (download dependencies)
- Create: `test/CMakeLists.txt`

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(memory-layer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- llama.cpp as submodule ---
set(LLAMA_METAL ON CACHE BOOL "Enable Metal" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/llama.cpp)

# --- SQLite3 ---
add_library(sqlite3 STATIC deps/sqlite3.c)
target_include_directories(sqlite3 PUBLIC deps)

# --- Main target ---
add_executable(memory-layer
    src/main.cpp
    src/config.cpp
    src/memory_store.cpp
    src/embedding.cpp
    src/injector.cpp
    src/proxy.cpp
)

target_include_directories(memory-layer PRIVATE
    include
    deps
    deps/llama.cpp/include
)

target_link_libraries(memory-layer PRIVATE
    llama
    common
    sqlite3
)

# macOS frameworks for Metal
if(APPLE)
    target_link_libraries(memory-layer PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
endif()

# --- Tests ---
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
```

- [ ] **Step 2: Add llama.cpp as git submodule**

Run:
```bash
cd /Users/a470718/Project/secret/memory-layer
git submodule add https://github.com/ggerganov/llama.cpp.git deps/llama.cpp
cd deps/llama.cpp && git checkout master && cd ../..
```

- [ ] **Step 3: Download header-only dependencies**

Run:
```bash
cd /Users/a470718/Project/secret/memory-layer/deps
# cpp-httplib
curl -sL https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h -o httplib.h
# nlohmann/json
curl -sL https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o json.hpp
# SQLite amalgamation
curl -sL https://www.sqlite.org/2024/sqlite-amalgamation-3450000.zip -o sqlite.zip
unzip -j sqlite.zip "*/sqlite3.c" "*/sqlite3.h" -d . && rm sqlite.zip
```

- [ ] **Step 4: Create test CMakeLists.txt**

Create `test/CMakeLists.txt`:
```cmake
# Minimal test framework (no external dep — just assert + main)
add_executable(test_memory_store test_memory_store.cpp ../src/memory_store.cpp)
target_include_directories(test_memory_store PRIVATE ../include ../deps)
target_link_libraries(test_memory_store PRIVATE sqlite3)

add_executable(test_injector test_injector.cpp ../src/injector.cpp)
target_include_directories(test_injector PRIVATE ../include ../deps)

add_executable(test_config test_config.cpp ../src/config.cpp)
target_include_directories(test_config PRIVATE ../include ../deps)

add_test(NAME test_memory_store COMMAND test_memory_store)
add_test(NAME test_injector COMMAND test_injector)
add_test(NAME test_config COMMAND test_config)
```

- [ ] **Step 5: Create placeholder source files and verify build**

Create minimal placeholders for all source files so CMake can configure:

`include/memorylayer/config.h`:
```cpp
#pragma once
#include <string>
#include <cstdint>

namespace memorylayer {

struct Config {
    std::string backend_url = "http://localhost:8080";
    int port = 8800;
    std::string embedding_model_path;
    std::string db_path = "memories.sqlite";
    int top_k = 5;
    int decay_days = 30;
    float dedup_threshold = 0.92f;
    int max_memories_per_agent = 1000;
    int max_memories_global = 5000;
    int gpu_layers = 99;
    float min_score_threshold = 0.3f;
    float agent_boost = 1.2f;
};

Config parse_args(int argc, char* argv[]);

} // namespace memorylayer
```

`include/memorylayer/memory_store.h`:
```cpp
#pragma once
namespace memorylayer { class MemoryStore {}; }
```

`include/memorylayer/embedding.h`:
```cpp
#pragma once
namespace memorylayer { class EmbeddingWorker {}; }
```

`include/memorylayer/injector.h`:
```cpp
#pragma once
namespace memorylayer {}
```

`include/memorylayer/proxy.h`:
```cpp
#pragma once
namespace memorylayer { class MemoryProxy {}; }
```

`src/main.cpp`:
```cpp
#include "memorylayer/config.h"
int main(int argc, char* argv[]) {
    auto cfg = memorylayer::parse_args(argc, argv);
    return 0;
}
```

`src/config.cpp`, `src/memory_store.cpp`, `src/embedding.cpp`, `src/injector.cpp`, `src/proxy.cpp`: empty stubs with the namespace.

Run:
```bash
cd /Users/a470718/Project/secret/memory-layer
mkdir -p build && cd build && cmake .. && make -j$(nproc) 2>&1 | tail -5
```
Expected: Build succeeds (possibly with warnings about unused code).

- [ ] **Step 6: Commit scaffolding**

```bash
echo "build/" >> .gitignore
echo "*.sqlite" >> .gitignore
git add -A && git commit -m "feat: project scaffolding with CMake + deps"
```

---

## Task 2: Config Module

**Files:**
- Modify: `include/memorylayer/config.h`
- Create: `src/config.cpp`
- Create: `test/test_config.cpp`

- [ ] **Step 1: Write failing test for config parsing**

Create `test/test_config.cpp`:
```cpp
#include "memorylayer/config.h"
#include <cassert>
#include <cstring>
#include <iostream>

void test_defaults() {
    const char* argv[] = {"memory-layer", "--embedding-model", "test.gguf"};
    auto cfg = memorylayer::parse_args(3, const_cast<char**>(argv));
    assert(cfg.backend_url == "http://localhost:8080");
    assert(cfg.port == 8800);
    assert(cfg.embedding_model_path == "test.gguf");
    assert(cfg.db_path == "memories.sqlite");
    assert(cfg.top_k == 5);
    assert(cfg.decay_days == 30);
    assert(cfg.gpu_layers == 99);
    std::cout << "test_defaults PASSED\n";
}

void test_custom_values() {
    const char* argv[] = {
        "memory-layer",
        "--backend", "http://myserver:9090",
        "--port", "9999",
        "--embedding-model", "custom.gguf",
        "--db", "custom.db",
        "--top-k", "3",
        "--decay-days", "7",
        "--dedup-threshold", "0.85",
        "--max-memories-per-agent", "500",
        "--gpu-layers", "32"
    };
    auto cfg = memorylayer::parse_args(21, const_cast<char**>(argv));
    assert(cfg.backend_url == "http://myserver:9090");
    assert(cfg.port == 9999);
    assert(cfg.embedding_model_path == "custom.gguf");
    assert(cfg.db_path == "custom.db");
    assert(cfg.top_k == 3);
    assert(cfg.decay_days == 7);
    assert(cfg.dedup_threshold == 0.85f);
    assert(cfg.max_memories_per_agent == 500);
    assert(cfg.gpu_layers == 32);
    std::cout << "test_custom_values PASSED\n";
}

int main() {
    test_defaults();
    test_custom_values();
    std::cout << "All config tests PASSED\n";
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && make test_config && ./test/test_config`
Expected: FAIL (link error or runtime assert because parse_args is not implemented)

- [ ] **Step 3: Implement config.cpp**

Create `src/config.cpp`:
```cpp
#include "memorylayer/config.h"
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace memorylayer {

static void print_usage() {
    std::cerr << "Usage: memory-layer --embedding-model <path> [options]\n"
              << "\nRequired:\n"
              << "  --embedding-model <path>    Path to GGUF embedding model\n"
              << "\nOptions:\n"
              << "  --backend <url>             LLM backend URL (default: http://localhost:8080)\n"
              << "  --port <n>                  Proxy listen port (default: 8800)\n"
              << "  --db <path>                 SQLite database path (default: memories.sqlite)\n"
              << "  --top-k <n>                 Memories to inject (default: 5)\n"
              << "  --decay-days <n>            Temporal decay half-life (default: 30)\n"
              << "  --dedup-threshold <f>       Deduplication cosine threshold (default: 0.92)\n"
              << "  --max-memories-per-agent <n> Max memories per agent (default: 1000)\n"
              << "  --gpu-layers <n>            GPU layers for embedding model (default: 99)\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (strcmp(arg, "--backend") == 0 && i + 1 < argc) {
            cfg.backend_url = argv[++i];
        } else if (strcmp(arg, "--port") == 0 && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--embedding-model") == 0 && i + 1 < argc) {
            cfg.embedding_model_path = argv[++i];
        } else if (strcmp(arg, "--db") == 0 && i + 1 < argc) {
            cfg.db_path = argv[++i];
        } else if (strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            cfg.top_k = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--decay-days") == 0 && i + 1 < argc) {
            cfg.decay_days = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--dedup-threshold") == 0 && i + 1 < argc) {
            cfg.dedup_threshold = std::atof(argv[++i]);
        } else if (strcmp(arg, "--max-memories-per-agent") == 0 && i + 1 < argc) {
            cfg.max_memories_per_agent = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--gpu-layers") == 0 && i + 1 < argc) {
            cfg.gpu_layers = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage();
            std::exit(0);
        }
    }

    if (cfg.embedding_model_path.empty()) {
        std::cerr << "Error: --embedding-model is required\n\n";
        print_usage();
        std::exit(1);
    }

    return cfg;
}

} // namespace memorylayer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd build && cmake .. && make test_config && ./test/test_config`
Expected: "All config tests PASSED"

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(config): CLI argument parsing with defaults"
```

---

## Task 3: Memory Store — Schema + Basic CRUD

**Files:**
- Modify: `include/memorylayer/memory_store.h`
- Create: `src/memory_store.cpp`
- Create: `test/test_memory_store.cpp`

- [ ] **Step 1: Define MemoryStore header**

Replace `include/memorylayer/memory_store.h`:
```cpp
#pragma once
#include "memorylayer/config.h"
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

struct sqlite3;

namespace memorylayer {

struct Memory {
    int64_t id;
    std::string agent_id;   // empty string = global
    double created_at;
    double updated_at;
    std::string user_text;
    std::string assist_text;
    std::vector<float> user_emb;   // dim=768
    std::vector<float> assist_emb; // dim=768
    int access_count;
};

struct ScoredMemory {
    Memory memory;
    float score;
};

class MemoryStore {
public:
    explicit MemoryStore(const std::string& db_path, int max_per_agent = 1000, int max_global = 5000);
    ~MemoryStore();

    // Disable copy
    MemoryStore(const MemoryStore&) = delete;
    MemoryStore& operator=(const MemoryStore&) = delete;

    // Insert a new memory. Returns the row id.
    int64_t insert(const std::string& agent_id,
                   const std::string& user_text,
                   const std::string& assist_text,
                   const std::vector<float>& user_emb,
                   const std::vector<float>& assist_emb);

    // Search for top-K relevant memories using dual-embedding cosine similarity.
    // Searches both agent-specific and global memories.
    std::vector<ScoredMemory> search(const std::vector<float>& query_emb,
                                     const std::string& agent_id,
                                     int top_k,
                                     float min_score,
                                     int decay_days,
                                     float agent_boost) const;

    // Find a duplicate (cosine > threshold). Returns the memory id or -1.
    int64_t find_duplicate(const std::vector<float>& user_emb, float threshold) const;

    // Update existing memory's timestamp and access count (for dedup merge).
    void touch(int64_t memory_id);

    // Evict oldest memories beyond the limit (FIFO).
    void evict(const std::string& agent_id);

    // Count memories for an agent (empty = global).
    int count(const std::string& agent_id) const;

private:
    sqlite3* db_ = nullptr;
    int max_per_agent_;
    int max_global_;

    void init_schema();
};

} // namespace memorylayer
```

- [ ] **Step 2: Write failing tests for insert + count**

Create `test/test_memory_store.cpp`:
```cpp
#include "memorylayer/memory_store.h"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <vector>

static std::vector<float> make_emb(float val) {
    return std::vector<float>(768, val);
}

void test_insert_and_count() {
    std::remove("test_store.sqlite");
    memorylayer::MemoryStore store("test_store.sqlite");

    assert(store.count("agent1") == 0);
    assert(store.count("") == 0);

    store.insert("agent1", "hello", "world", make_emb(0.1f), make_emb(0.2f));
    assert(store.count("agent1") == 1);
    assert(store.count("") == 0);

    store.insert("", "global q", "global a", make_emb(0.3f), make_emb(0.4f));
    assert(store.count("") == 1);

    store.insert("agent1", "second", "response", make_emb(0.5f), make_emb(0.6f));
    assert(store.count("agent1") == 2);

    std::remove("test_store.sqlite");
    std::cout << "test_insert_and_count PASSED\n";
}

void test_eviction() {
    std::remove("test_evict.sqlite");
    memorylayer::MemoryStore store("test_evict.sqlite", 3, 5000);

    for (int i = 0; i < 5; i++) {
        store.insert("agent1", "q" + std::to_string(i), "a" + std::to_string(i),
                     make_emb(0.1f * i), make_emb(0.2f * i));
    }
    // Before eviction, count is 5
    assert(store.count("agent1") == 5);

    store.evict("agent1");
    // After eviction with max=3, count should be 3
    assert(store.count("agent1") == 3);

    std::remove("test_evict.sqlite");
    std::cout << "test_eviction PASSED\n";
}

int main() {
    test_insert_and_count();
    test_eviction();
    std::cout << "All memory_store tests PASSED\n";
    return 0;
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd build && cmake .. && make test_memory_store 2>&1 | tail -10`
Expected: Compilation fails (MemoryStore methods not implemented)

- [ ] **Step 4: Implement memory_store.cpp (schema + insert + count + evict)**

Create `src/memory_store.cpp`:
```cpp
#include "memorylayer/memory_store.h"
#include "sqlite3.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <ctime>

namespace memorylayer {

MemoryStore::MemoryStore(const std::string& db_path, int max_per_agent, int max_global)
    : max_per_agent_(max_per_agent), max_global_(max_global) {
    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to open database: " + std::string(sqlite3_errmsg(db_)));
    }
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
    init_schema();
}

MemoryStore::~MemoryStore() {
    if (db_) sqlite3_close(db_);
}

void MemoryStore::init_schema() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id     TEXT,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL,
            user_text    TEXT NOT NULL,
            assist_text  TEXT NOT NULL,
            user_emb     BLOB NOT NULL,
            assist_emb   BLOB NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
    )";
    char* err = nullptr;
    sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (err) {
        std::string msg = err;
        sqlite3_free(err);
        throw std::runtime_error("Schema init failed: " + msg);
    }
}

static double now_unix() {
    return static_cast<double>(std::time(nullptr));
}

int64_t MemoryStore::insert(const std::string& agent_id,
                            const std::string& user_text,
                            const std::string& assist_text,
                            const std::vector<float>& user_emb,
                            const std::vector<float>& assist_emb) {
    const char* sql = R"(
        INSERT INTO memories (agent_id, created_at, updated_at, user_text, assist_text, user_emb, assist_emb)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    // agent_id: NULL for global, text otherwise
    if (agent_id.empty()) {
        sqlite3_bind_null(stmt, 1);
    } else {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
    }

    double ts = now_unix();
    sqlite3_bind_double(stmt, 2, ts);
    sqlite3_bind_double(stmt, 3, ts);
    sqlite3_bind_text(stmt, 4, user_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, assist_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 6, user_emb.data(), user_emb.size() * sizeof(float), SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 7, assist_emb.data(), assist_emb.size() * sizeof(float), SQLITE_TRANSIENT);

    sqlite3_step(stmt);
    int64_t id = sqlite3_last_insert_rowid(db_);
    sqlite3_finalize(stmt);
    return id;
}

int MemoryStore::count(const std::string& agent_id) const {
    const char* sql = agent_id.empty()
        ? "SELECT COUNT(*) FROM memories WHERE agent_id IS NULL"
        : "SELECT COUNT(*) FROM memories WHERE agent_id = ?";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (!agent_id.empty()) {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
    }

    int result = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return result;
}

void MemoryStore::touch(int64_t memory_id) {
    const char* sql = "UPDATE memories SET updated_at = ?, access_count = access_count + 1 WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_double(stmt, 1, now_unix());
    sqlite3_bind_int64(stmt, 2, memory_id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

void MemoryStore::evict(const std::string& agent_id) {
    int limit = agent_id.empty() ? max_global_ : max_per_agent_;
    int current = count(agent_id);
    if (current <= limit) return;

    int to_delete = current - limit;

    const char* sql = agent_id.empty()
        ? "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE agent_id IS NULL ORDER BY created_at ASC LIMIT ?)"
        : "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?)";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (agent_id.empty()) {
        sqlite3_bind_int(stmt, 1, to_delete);
    } else {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 2, to_delete);
    }
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

// Cosine similarity between two float vectors
static float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0.0f ? dot / denom : 0.0f;
}

std::vector<ScoredMemory> MemoryStore::search(const std::vector<float>& query_emb,
                                               const std::string& agent_id,
                                               int top_k,
                                               float min_score,
                                               int decay_days,
                                               float agent_boost) const {
    // Load all memories for this agent + global
    const char* sql = agent_id.empty()
        ? "SELECT id, agent_id, created_at, updated_at, user_text, assist_text, user_emb, assist_emb, access_count FROM memories WHERE agent_id IS NULL"
        : "SELECT id, agent_id, created_at, updated_at, user_text, assist_text, user_emb, assist_emb, access_count FROM memories WHERE agent_id IS NULL OR agent_id = ?";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (!agent_id.empty()) {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
    }

    double now = now_unix();
    double decay_hours_max = 24.0 * decay_days;
    int dim = static_cast<int>(query_emb.size());

    std::vector<ScoredMemory> results;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Memory mem;
        mem.id = sqlite3_column_int64(stmt, 0);

        const char* aid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        mem.agent_id = aid ? aid : "";
        mem.created_at = sqlite3_column_double(stmt, 2);
        mem.updated_at = sqlite3_column_double(stmt, 3);

        const char* ut = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        const char* at = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        mem.user_text = ut ? ut : "";
        mem.assist_text = at ? at : "";

        const float* u_blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 6));
        int u_size = sqlite3_column_bytes(stmt, 6) / sizeof(float);
        mem.user_emb.assign(u_blob, u_blob + u_size);

        const float* a_blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 7));
        int a_size = sqlite3_column_bytes(stmt, 7) / sizeof(float);
        mem.assist_emb.assign(a_blob, a_blob + a_size);

        mem.access_count = sqlite3_column_int(stmt, 8);

        // Dual cosine: max of user_emb match and assist_emb match
        float cos_user = cosine_similarity(query_emb.data(), mem.user_emb.data(), dim);
        float cos_assist = cosine_similarity(query_emb.data(), mem.assist_emb.data(), dim);
        float cos_max = std::max(cos_user, cos_assist);

        // Temporal decay
        double age_hours = (now - mem.created_at) / 3600.0;
        float decay = std::max(0.5f, 1.0f - static_cast<float>(age_hours / decay_hours_max));

        float score = cos_max * decay;

        // Agent boost
        if (!mem.agent_id.empty() && mem.agent_id == agent_id) {
            score *= agent_boost;
        }

        if (score >= min_score) {
            results.push_back({std::move(mem), score});
        }
    }
    sqlite3_finalize(stmt);

    // Sort descending by score, take top_k
    std::sort(results.begin(), results.end(),
              [](const ScoredMemory& a, const ScoredMemory& b) { return a.score > b.score; });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }

    return results;
}

int64_t MemoryStore::find_duplicate(const std::vector<float>& user_emb, float threshold) const {
    const char* sql = "SELECT id, user_emb FROM memories";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    int dim = static_cast<int>(user_emb.size());
    int64_t best_id = -1;
    float best_cos = 0.0f;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t id = sqlite3_column_int64(stmt, 0);
        const float* blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 1));
        int size = sqlite3_column_bytes(stmt, 1) / sizeof(float);
        if (size != dim) continue;

        float cos = cosine_similarity(user_emb.data(), blob, dim);
        if (cos > best_cos) {
            best_cos = cos;
            best_id = id;
        }
    }
    sqlite3_finalize(stmt);

    return best_cos >= threshold ? best_id : -1;
}

} // namespace memorylayer
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd build && cmake .. && make test_memory_store && ./test/test_memory_store`
Expected: "All memory_store tests PASSED"

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(store): MemoryStore with SQLite CRUD, search, dedup, eviction"
```

---

## Task 4: Cosine Similarity Tests

**Files:**
- Create: `test/test_cosine.cpp`

- [ ] **Step 1: Write cosine similarity math tests**

Create `test/test_cosine.cpp`:
```cpp
#include "memorylayer/memory_store.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>

static std::vector<float> make_vec(std::initializer_list<float> vals, int pad_to = 768) {
    std::vector<float> v(vals);
    v.resize(pad_to, 0.0f);
    return v;
}

void test_search_returns_most_similar() {
    std::remove("test_cosine.sqlite");
    memorylayer::MemoryStore store("test_cosine.sqlite");

    // Insert 3 memories with distinct directions
    auto emb_a = make_vec({1.0f, 0.0f, 0.0f});
    auto emb_b = make_vec({0.0f, 1.0f, 0.0f});
    auto emb_c = make_vec({0.7f, 0.7f, 0.0f}); // between a and b

    store.insert("", "question A", "answer A", emb_a, emb_a);
    store.insert("", "question B", "answer B", emb_b, emb_b);
    store.insert("", "question C", "answer C", emb_c, emb_c);

    // Query close to A
    auto query = make_vec({0.9f, 0.1f, 0.0f});
    auto results = store.search(query, "", 2, 0.0f, 9999, 1.0f);

    assert(results.size() == 2);
    // First result should be A (most similar to query)
    assert(results[0].memory.user_text == "question A");
    // Second should be C (partial match)
    assert(results[1].memory.user_text == "question C");
    assert(results[0].score > results[1].score);

    std::remove("test_cosine.sqlite");
    std::cout << "test_search_returns_most_similar PASSED\n";
}

void test_dual_embedding_matches_assist() {
    std::remove("test_dual.sqlite");
    memorylayer::MemoryStore store("test_dual.sqlite");

    // Memory where assist_emb is closer to query than user_emb
    auto user_emb = make_vec({1.0f, 0.0f, 0.0f});
    auto assist_emb = make_vec({0.0f, 1.0f, 0.0f});
    store.insert("", "unrelated question", "relevant answer", user_emb, assist_emb);

    // Query aligned with assist direction
    auto query = make_vec({0.0f, 0.9f, 0.1f});
    auto results = store.search(query, "", 5, 0.0f, 9999, 1.0f);

    assert(results.size() == 1);
    assert(results[0].score > 0.5f); // matched via assist_emb
    std::cout << "test_dual_embedding_matches_assist PASSED\n";

    std::remove("test_dual.sqlite");
}

void test_dedup_finds_similar() {
    std::remove("test_dedup.sqlite");
    memorylayer::MemoryStore store("test_dedup.sqlite");

    auto emb = make_vec({1.0f, 0.0f, 0.0f});
    store.insert("", "hello", "world", emb, emb);

    // Slightly different vector
    auto similar = make_vec({0.99f, 0.01f, 0.0f});
    int64_t dup = store.find_duplicate(similar, 0.99f);
    assert(dup > 0); // found duplicate

    // Very different vector
    auto different = make_vec({0.0f, 1.0f, 0.0f});
    int64_t no_dup = store.find_duplicate(different, 0.99f);
    assert(no_dup == -1); // no duplicate

    std::remove("test_dedup.sqlite");
    std::cout << "test_dedup_finds_similar PASSED\n";
}

int main() {
    test_search_returns_most_similar();
    test_dual_embedding_matches_assist();
    test_dedup_finds_similar();
    std::cout << "All cosine tests PASSED\n";
    return 0;
}
```

- [ ] **Step 2: Add to test CMakeLists and run**

Add to `test/CMakeLists.txt`:
```cmake
add_executable(test_cosine test_cosine.cpp ../src/memory_store.cpp)
target_include_directories(test_cosine PRIVATE ../include ../deps)
target_link_libraries(test_cosine PRIVATE sqlite3)
add_test(NAME test_cosine COMMAND test_cosine)
```

Run: `cd build && cmake .. && make test_cosine && ./test/test_cosine`
Expected: "All cosine tests PASSED"

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test(store): cosine similarity, dual-embedding, dedup verification"
```

---

## Task 5: Injector Module

**Files:**
- Modify: `include/memorylayer/injector.h`
- Create: `src/injector.cpp`
- Create: `test/test_injector.cpp`

- [ ] **Step 1: Define injector header**

Replace `include/memorylayer/injector.h`:
```cpp
#pragma once
#include "memorylayer/memory_store.h"
#include "json.hpp"
#include <string>
#include <vector>

namespace memorylayer {

// Format a scored memory into a single-line summary string.
// Truncates user_text and assist_text to max_chars each.
std::string format_memory_line(const ScoredMemory& mem, double now_unix, int max_chars = 200);

// Format multiple memories into the <memory context> block.
std::string format_memory_context(const std::vector<ScoredMemory>& memories, double now_unix);

// Inject memory context into the messages array (modifies in place).
// If no system message exists, prepends one. If one exists, appends to it.
void inject_memories(nlohmann::json& messages, const std::string& memory_context);

} // namespace memorylayer
```

- [ ] **Step 2: Write failing tests**

Create `test/test_injector.cpp`:
```cpp
#include "memorylayer/injector.h"
#include <cassert>
#include <iostream>
#include <ctime>

using json = nlohmann::json;

void test_format_memory_line() {
    memorylayer::ScoredMemory sm;
    sm.memory.user_text = "How do I sort a vector in C++?";
    sm.memory.assist_text = "Use std::sort with iterators. Example: std::sort(v.begin(), v.end())";
    sm.memory.created_at = static_cast<double>(std::time(nullptr)) - 7200; // 2h ago
    sm.score = 0.85f;

    double now = static_cast<double>(std::time(nullptr));
    auto line = memorylayer::format_memory_line(sm, now);

    assert(line.find("[2h ago]") != std::string::npos);
    assert(line.find("sort a vector") != std::string::npos);
    assert(line.find("std::sort") != std::string::npos);
    std::cout << "test_format_memory_line PASSED\n";
}

void test_format_memory_context_empty() {
    std::vector<memorylayer::ScoredMemory> empty;
    auto ctx = memorylayer::format_memory_context(empty, 0.0);
    assert(ctx.empty());
    std::cout << "test_format_memory_context_empty PASSED\n";
}

void test_inject_with_existing_system() {
    json messages = json::array({
        {{"role", "system"}, {"content", "You are a helpful assistant."}},
        {{"role", "user"}, {"content", "What is C++?"}}
    });

    std::string ctx = "<memory context>\nRelevant past interactions:\n- [1h ago] test memory\n</memory context>";
    memorylayer::inject_memories(messages, ctx);

    std::string sys = messages[0]["content"].get<std::string>();
    assert(sys.find("You are a helpful assistant.") != std::string::npos);
    assert(sys.find("<memory context>") != std::string::npos);
    std::cout << "test_inject_with_existing_system PASSED\n";
}

void test_inject_without_system() {
    json messages = json::array({
        {{"role", "user"}, {"content", "Hello"}}
    });

    std::string ctx = "<memory context>\ntest\n</memory context>";
    memorylayer::inject_memories(messages, ctx);

    assert(messages.size() == 2); // system was prepended
    assert(messages[0]["role"] == "system");
    assert(messages[0]["content"].get<std::string>().find("<memory context>") != std::string::npos);
    assert(messages[1]["role"] == "user");
    std::cout << "test_inject_without_system PASSED\n";
}

void test_no_injection_on_empty_context() {
    json messages = json::array({
        {{"role", "user"}, {"content", "Hello"}}
    });

    memorylayer::inject_memories(messages, "");
    assert(messages.size() == 1); // unchanged
    std::cout << "test_no_injection_on_empty_context PASSED\n";
}

int main() {
    test_format_memory_line();
    test_format_memory_context_empty();
    test_inject_with_existing_system();
    test_inject_without_system();
    test_no_injection_on_empty_context();
    std::cout << "All injector tests PASSED\n";
    return 0;
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd build && cmake .. && make test_injector 2>&1 | tail -5`
Expected: Link error (functions not implemented)

- [ ] **Step 4: Implement injector.cpp**

Create `src/injector.cpp`:
```cpp
#include "memorylayer/injector.h"
#include <cmath>
#include <sstream>

namespace memorylayer {

static std::string format_time_ago(double seconds_ago) {
    if (seconds_ago < 60) return "just now";
    if (seconds_ago < 3600) return std::to_string(static_cast<int>(seconds_ago / 60)) + "m ago";
    if (seconds_ago < 86400) return std::to_string(static_cast<int>(seconds_ago / 3600)) + "h ago";
    return std::to_string(static_cast<int>(seconds_ago / 86400)) + "d ago";
}

static std::string truncate(const std::string& s, int max_chars) {
    if (static_cast<int>(s.size()) <= max_chars) return s;
    return s.substr(0, max_chars) + "...";
}

std::string format_memory_line(const ScoredMemory& mem, double now_unix, int max_chars) {
    double age_seconds = now_unix - mem.memory.created_at;
    std::string time_str = format_time_ago(age_seconds);
    std::string user_part = truncate(mem.memory.user_text, max_chars);
    std::string assist_part = truncate(mem.memory.assist_text, max_chars);
    return "- [" + time_str + "] " + user_part + " Response: " + assist_part;
}

std::string format_memory_context(const std::vector<ScoredMemory>& memories, double now_unix) {
    if (memories.empty()) return "";

    std::ostringstream oss;
    oss << "<memory context>\nRelevant past interactions:\n";
    for (const auto& sm : memories) {
        oss << format_memory_line(sm, now_unix) << "\n";
    }
    oss << "</memory context>";
    return oss.str();
}

void inject_memories(nlohmann::json& messages, const std::string& memory_context) {
    if (memory_context.empty()) return;

    // Check if first message is system
    if (!messages.empty() && messages[0].value("role", "") == "system") {
        std::string content = messages[0]["content"].get<std::string>();
        content += "\n\n" + memory_context;
        messages[0]["content"] = content;
    } else {
        // Prepend a new system message
        nlohmann::json sys_msg = {{"role", "system"}, {"content", memory_context}};
        messages.insert(messages.begin(), sys_msg);
    }
}

} // namespace memorylayer
```

- [ ] **Step 5: Update test CMakeLists.txt**

Replace the `test_injector` target in `test/CMakeLists.txt`:
```cmake
add_executable(test_injector test_injector.cpp ../src/injector.cpp)
target_include_directories(test_injector PRIVATE ../include ../deps)
target_link_libraries(test_injector PRIVATE)
add_test(NAME test_injector COMMAND test_injector)
```

- [ ] **Step 6: Run tests**

Run: `cd build && cmake .. && make test_injector && ./test/test_injector`
Expected: "All injector tests PASSED"

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(injector): memory formatting and message injection"
```

---

## Task 6: Embedding Worker

**Files:**
- Modify: `include/memorylayer/embedding.h`
- Create: `src/embedding.cpp`

- [ ] **Step 1: Define embedding worker header**

Replace `include/memorylayer/embedding.h`:
```cpp
#pragma once
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <atomic>

namespace memorylayer {

class EmbeddingWorker {
public:
    // Initialize with model path and GPU layers. Loads model on construction.
    EmbeddingWorker(const std::string& model_path, int gpu_layers);
    ~EmbeddingWorker();

    // Disable copy
    EmbeddingWorker(const EmbeddingWorker&) = delete;
    EmbeddingWorker& operator=(const EmbeddingWorker&) = delete;

    // Submit text for embedding. Blocks until result is ready.
    // Thread-safe: can be called from any thread.
    std::vector<float> embed(const std::string& text);

    // Get embedding dimension
    int dimension() const { return n_embd_; }

    // Check if model loaded successfully
    bool is_ready() const { return ready_; }

    // Shutdown the worker thread
    void shutdown();

private:
    struct EmbedJob {
        std::string text;
        std::promise<std::vector<float>> promise;
    };

    void worker_loop();

    // llama.cpp handles (forward-declared to avoid header pollution)
    void* model_ = nullptr;  // llama_model*
    void* ctx_ = nullptr;    // llama_context*
    int n_embd_ = 0;
    bool ready_ = false;

    std::thread worker_thread_;
    std::queue<EmbedJob> jobs_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
};

} // namespace memorylayer
```

- [ ] **Step 2: Implement embedding.cpp**

Create `src/embedding.cpp`:
```cpp
#include "memorylayer/embedding.h"
#include "llama.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <numeric>

namespace memorylayer {

EmbeddingWorker::EmbeddingWorker(const std::string& model_path, int gpu_layers) {
    // Initialize llama backend
    llama_backend_init();

    // Load model
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;

    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "[embedding] Failed to load model: " << model_path << "\n";
        return;
    }
    model_ = model;

    // Create context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;  // embedding models don't need large context
    ctx_params.n_batch = 2048;
    ctx_params.embeddings = true;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "[embedding] Failed to create context\n";
        llama_model_free(model);
        model_ = nullptr;
        return;
    }
    ctx_ = ctx;

    n_embd_ = llama_model_n_embd(model);
    ready_ = true;

    std::cout << "[embedding] Model loaded: " << model_path
              << " (dim=" << n_embd_ << ", gpu_layers=" << gpu_layers << ")\n";

    // Start worker thread
    worker_thread_ = std::thread(&EmbeddingWorker::worker_loop, this);
}

EmbeddingWorker::~EmbeddingWorker() {
    shutdown();
    if (ctx_) llama_free(static_cast<llama_context*>(ctx_));
    if (model_) llama_model_free(static_cast<llama_model*>(model_));
    llama_backend_free();
}

void EmbeddingWorker::shutdown() {
    stop_ = true;
    cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

std::vector<float> EmbeddingWorker::embed(const std::string& text) {
    if (!ready_) return {};

    std::promise<std::vector<float>> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.push({text, std::move(promise)});
    }
    cv_.notify_one();

    return future.get();
}

void EmbeddingWorker::worker_loop() {
    auto* model = static_cast<llama_model*>(model_);
    auto* ctx = static_cast<llama_context*>(ctx_);
    const auto* vocab = llama_model_get_vocab(model);

    while (!stop_) {
        EmbedJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !jobs_.empty() || stop_; });
            if (stop_ && jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop();
        }

        // Tokenize
        int max_tokens = job.text.size() + 32;
        std::vector<llama_token> tokens(max_tokens);
        int n_tokens = llama_tokenize(vocab, job.text.c_str(), job.text.size(),
                                      tokens.data(), max_tokens, true, true);
        if (n_tokens < 0) {
            job.promise.set_value({});
            continue;
        }
        tokens.resize(n_tokens);

        // Clear KV cache
        llama_kv_cache_clear(ctx);

        // Create batch
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            llama_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
        }

        // Decode
        if (llama_decode(ctx, batch) != 0) {
            job.promise.set_value({});
            llama_batch_free(batch);
            continue;
        }

        // Get embeddings (mean pooling over sequence)
        std::vector<float> result(n_embd_, 0.0f);
        const float* emb = llama_get_embeddings_seq(ctx, 0);
        if (emb) {
            std::memcpy(result.data(), emb, n_embd_ * sizeof(float));
        } else {
            // Fallback: get embedding from last token
            const float* last_emb = llama_get_embeddings_ith(ctx, n_tokens - 1);
            if (last_emb) {
                std::memcpy(result.data(), last_emb, n_embd_ * sizeof(float));
            }
        }

        // L2 normalize
        float norm = 0.0f;
        for (float v : result) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& v : result) v /= norm;
        }

        llama_batch_free(batch);
        job.promise.set_value(std::move(result));
    }
}

} // namespace memorylayer
```

- [ ] **Step 3: Verify compilation**

Run: `cd build && cmake .. && make memory-layer 2>&1 | tail -10`
Expected: Build succeeds (no test for embedding — requires a real model file)

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(embedding): llama.cpp worker thread with Metal support"
```

---

## Task 7: HTTP Proxy

**Files:**
- Modify: `include/memorylayer/proxy.h`
- Create: `src/proxy.cpp`

- [ ] **Step 1: Define proxy header**

Replace `include/memorylayer/proxy.h`:
```cpp
#pragma once
#include "memorylayer/config.h"
#include "memorylayer/memory_store.h"
#include "memorylayer/embedding.h"
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace memorylayer {

struct SaveJob {
    std::string agent_id;
    std::string user_text;
    std::string assist_text;
};

class MemoryProxy {
public:
    MemoryProxy(const Config& cfg, MemoryStore& store, EmbeddingWorker& embedder);
    ~MemoryProxy();

    // Start the HTTP server (blocks until shutdown)
    void run();

    // Signal shutdown
    void shutdown();

private:
    const Config& cfg_;
    MemoryStore& store_;
    EmbeddingWorker& embedder_;

    std::atomic<bool> stop_{false};

    // Background saver thread
    std::thread saver_thread_;
    std::queue<SaveJob> save_queue_;
    std::mutex save_mutex_;
    std::condition_variable save_cv_;

    void saver_loop();
    void enqueue_save(const std::string& agent_id,
                      const std::string& user_text,
                      const std::string& assist_text);

    // Core logic
    std::string retrieve_and_inject(const std::string& agent_id,
                                    const std::string& user_text,
                                    nlohmann::json& messages);

    std::string extract_last_user_message(const nlohmann::json& messages) const;
};

} // namespace memorylayer
```

- [ ] **Step 2: Implement proxy.cpp**

Create `src/proxy.cpp`:
```cpp
#include "memorylayer/proxy.h"
#include "memorylayer/injector.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"
#include "json.hpp"
#include <iostream>
#include <sstream>
#include <ctime>

using json = nlohmann::json;

namespace memorylayer {

MemoryProxy::MemoryProxy(const Config& cfg, MemoryStore& store, EmbeddingWorker& embedder)
    : cfg_(cfg), store_(store), embedder_(embedder) {
    saver_thread_ = std::thread(&MemoryProxy::saver_loop, this);
}

MemoryProxy::~MemoryProxy() {
    shutdown();
}

void MemoryProxy::shutdown() {
    stop_ = true;
    save_cv_.notify_all();
    if (saver_thread_.joinable()) saver_thread_.join();
}

std::string MemoryProxy::extract_last_user_message(const json& messages) const {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if ((*it).value("role", "") == "user") {
            return (*it).value("content", "");
        }
    }
    return "";
}

std::string MemoryProxy::retrieve_and_inject(const std::string& agent_id,
                                              const std::string& user_text,
                                              json& messages) {
    if (user_text.empty() || !embedder_.is_ready()) return "";

    // Embed the user query
    auto query_emb = embedder_.embed(user_text);
    if (query_emb.empty()) return "";

    // Search for relevant memories
    auto memories = store_.search(query_emb, agent_id, cfg_.top_k,
                                   cfg_.min_score_threshold, cfg_.decay_days,
                                   cfg_.agent_boost);

    if (memories.empty()) return "";

    // Format and inject
    double now = static_cast<double>(std::time(nullptr));
    std::string context = format_memory_context(memories, now);
    inject_memories(messages, context);

    return context;
}

void MemoryProxy::enqueue_save(const std::string& agent_id,
                                const std::string& user_text,
                                const std::string& assist_text) {
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        save_queue_.push({agent_id, user_text, assist_text});
    }
    save_cv_.notify_one();
}

void MemoryProxy::saver_loop() {
    while (!stop_) {
        SaveJob job;
        {
            std::unique_lock<std::mutex> lock(save_mutex_);
            save_cv_.wait(lock, [this] { return !save_queue_.empty() || stop_; });
            if (stop_ && save_queue_.empty()) return;
            job = std::move(save_queue_.front());
            save_queue_.pop();
        }

        if (!embedder_.is_ready()) continue;

        // Embed both user and assistant text
        auto user_emb = embedder_.embed(job.user_text);
        auto assist_emb = embedder_.embed(job.assist_text);
        if (user_emb.empty() || assist_emb.empty()) continue;

        // Check for duplicates
        int64_t dup_id = store_.find_duplicate(user_emb, cfg_.dedup_threshold);
        if (dup_id > 0) {
            store_.touch(dup_id);
            std::cout << "[memory] Deduplicated: merged with memory #" << dup_id << "\n";
        } else {
            store_.insert(job.agent_id, job.user_text, job.assist_text, user_emb, assist_emb);
            store_.evict(job.agent_id);
            std::cout << "[memory] Saved new memory for agent='"
                      << (job.agent_id.empty() ? "global" : job.agent_id) << "'\n";
        }
    }
}

void MemoryProxy::run() {
    httplib::Server svr;

    // Health endpoint
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Main proxy endpoint: /v1/chat/completions
    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        // Parse request
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid JSON"})", "application/json");
            return;
        }

        // Extract agent_id from header
        std::string agent_id;
        if (req.has_header("X-Agent-Id")) {
            agent_id = req.get_header_value("X-Agent-Id");
        }

        // Extract messages
        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing messages array"})", "application/json");
            return;
        }

        auto& messages = body["messages"];
        std::string user_text = extract_last_user_message(messages);

        // Retrieve and inject memories
        retrieve_and_inject(agent_id, user_text, messages);

        bool streaming = body.value("stream", false);

        // Forward to backend
        httplib::Client cli(cfg_.backend_url);
        cli.set_read_timeout(300);  // 5 min for long generations
        cli.set_connection_timeout(10);

        std::string modified_body = body.dump();

        if (!streaming) {
            // Non-streaming: forward, capture, return
            auto backend_res = cli.Post("/v1/chat/completions",
                                         modified_body, "application/json");
            if (!backend_res) {
                res.status = 502;
                res.set_content(R"({"error":"Backend unreachable"})", "application/json");
                return;
            }

            res.status = backend_res->status;
            res.set_content(backend_res->body, backend_res->get_header_value("Content-Type"));

            // Save memory in background (only on success)
            if (backend_res->status == 200 && !user_text.empty()) {
                try {
                    auto resp_json = json::parse(backend_res->body);
                    std::string assist_text = resp_json["choices"][0]["message"]["content"].get<std::string>();
                    enqueue_save(agent_id, user_text, assist_text);
                } catch (...) {
                    // Response parsing failed — skip saving
                }
            }
        } else {
            // Streaming: forward SSE chunks, buffer assistant content
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            std::string accumulated_content;

            // Use chunked transfer with content provider
            res.set_chunked_content_provider("text/event-stream",
                [&](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                    auto result = cli.Post("/v1/chat/completions",
                        modified_body, "application/json",
                        [&](const char* data, size_t len) -> bool {
                            // Forward each chunk immediately
                            sink.write(data, len);

                            // Parse SSE lines to accumulate content
                            std::string chunk(data, len);
                            std::istringstream stream(chunk);
                            std::string line;
                            while (std::getline(stream, line)) {
                                if (line.substr(0, 6) == "data: ") {
                                    std::string payload = line.substr(6);
                                    if (payload == "[DONE]") continue;
                                    try {
                                        auto j = json::parse(payload);
                                        auto delta = j["choices"][0]["delta"];
                                        if (delta.contains("content")) {
                                            accumulated_content += delta["content"].get<std::string>();
                                        }
                                    } catch (...) {}
                                }
                            }
                            return true; // continue receiving
                        });

                    if (!result) {
                        sink.write("data: {\"error\":\"backend unreachable\"}\n\n", 39);
                    }

                    sink.done();
                    return true;  // transfer complete
                });

            // Save accumulated content
            if (!user_text.empty() && !accumulated_content.empty()) {
                enqueue_save(agent_id, user_text, accumulated_content);
            }
        }
    });

    // Catch-all: forward any other /v1/* requests to backend unchanged
    svr.Post("/v1/.*", [this](const httplib::Request& req, httplib::Response& res) {
        httplib::Client cli(cfg_.backend_url);
        cli.set_read_timeout(60);
        auto result = cli.Post(req.path, req.body,
                               req.get_header_value("Content-Type"));
        if (result) {
            res.status = result->status;
            res.set_content(result->body, result->get_header_value("Content-Type"));
        } else {
            res.status = 502;
            res.set_content(R"({"error":"Backend unreachable"})", "application/json");
        }
    });

    std::cout << "[proxy] Listening on http://localhost:" << cfg_.port << "\n";
    std::cout << "[proxy] Backend: " << cfg_.backend_url << "\n";
    std::cout << "[proxy] Memory injection: top_k=" << cfg_.top_k
              << ", decay_days=" << cfg_.decay_days << "\n";

    svr.listen("0.0.0.0", cfg_.port);
}

} // namespace memorylayer
```

- [ ] **Step 3: Verify compilation**

Run: `cd build && cmake .. && make memory-layer 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(proxy): HTTP proxy with streaming, memory injection, background save"
```

---

## Task 8: Main Entry Point

**Files:**
- Modify: `src/main.cpp`

- [ ] **Step 1: Wire all components in main.cpp**

Replace `src/main.cpp`:
```cpp
#include "memorylayer/config.h"
#include "memorylayer/memory_store.h"
#include "memorylayer/embedding.h"
#include "memorylayer/proxy.h"
#include <iostream>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int) {
    g_shutdown = true;
    std::cout << "\n[main] Shutting down...\n";
}

int main(int argc, char* argv[]) {
    auto cfg = memorylayer::parse_args(argc, argv);

    std::cout << "=== Agent Memory Layer ===\n";
    std::cout << "[main] Embedding model: " << cfg.embedding_model_path << "\n";
    std::cout << "[main] Database: " << cfg.db_path << "\n";
    std::cout << "[main] Backend: " << cfg.backend_url << "\n";
    std::cout << "[main] Port: " << cfg.port << "\n";

    // Initialize components
    memorylayer::MemoryStore store(cfg.db_path, cfg.max_memories_per_agent, cfg.max_memories_global);
    std::cout << "[main] SQLite store initialized\n";

    memorylayer::EmbeddingWorker embedder(cfg.embedding_model_path, cfg.gpu_layers);
    if (!embedder.is_ready()) {
        std::cerr << "[main] ERROR: Failed to load embedding model\n";
        return 1;
    }
    std::cout << "[main] Embedding worker ready (dim=" << embedder.dimension() << ")\n";

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Run proxy (blocks until server stops)
    memorylayer::MemoryProxy proxy(cfg, store, embedder);
    proxy.run();

    return 0;
}
```

- [ ] **Step 2: Build and verify**

Run: `cd build && cmake .. && make memory-layer 2>&1 | tail -5`
Expected: Build succeeds, produces `./memory-layer` binary

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(main): entry point wiring all components together"
```

---

## Task 9: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `README.md`:
```markdown
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
- 📡 **Streaming support** — zero added latency on response streaming

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

## Architecture

- **C++17** single binary
- **llama.cpp** for local embedding inference (Metal/GPU)
- **SQLite** for persistent memory storage
- **cpp-httplib** for HTTP server + client
- **Dedicated threads**: embedding worker + background saver (non-blocking proxy)

## Recommended Embedding Model

[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) — 768 dimensions, excellent quality/speed ratio on Apple Silicon.

```bash
# Download from HuggingFace
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md && git commit -m "docs: comprehensive README with usage examples"
```

---

## Task 10: End-to-End Verification

- [ ] **Step 1: Full clean build**

Run:
```bash
cd /Users/a470718/Project/secret/memory-layer
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc) 2>&1 | tail -20
```
Expected: Build succeeds, `./memory-layer` binary produced

- [ ] **Step 2: Run unit tests**

Run: `cd build && ctest --output-on-failure`
Expected: All tests pass (config, memory_store, cosine, injector)

- [ ] **Step 3: Smoke test the binary**

Run: `./build/memory-layer --help`
Expected: Prints usage help and exits

- [ ] **Step 4: Final commit with .gitignore**

Verify `.gitignore` contains:
```
build/
*.sqlite
deps/llama.cpp/
```

```bash
git add -A && git commit -m "chore: final verification, gitignore cleanup"
```

---

## Dependency Graph

```
Task 1 (Scaffolding)
    ├── Task 2 (Config)
    ├── Task 3 (Memory Store CRUD)
    │       └── Task 4 (Cosine Tests)
    ├── Task 5 (Injector)
    └── Task 6 (Embedding Worker)
              └── Task 7 (Proxy) ← depends on Store + Embedding + Injector
                      └── Task 8 (Main) ← depends on all
                              └── Task 9 (README)
                                      └── Task 10 (E2E Verification)
```

Tasks 2, 3, 5, 6 are **parallelizable** after Task 1 completes.
