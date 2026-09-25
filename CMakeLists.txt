cmake_minimum_required(VERSION 3.20)
project(memory-layer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- llama.cpp as submodule ---
set(GGML_METAL ON CACHE BOOL "Enable Metal" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/llama.cpp)

# --- SQLite3 ---
add_library(sqlite3 STATIC deps/sqlite3.c)
target_include_directories(sqlite3 PUBLIC deps)
target_compile_options(sqlite3 PRIVATE -w)

# --- Main target ---
add_executable(memory-layer
    src/main.cpp
    src/config.cpp
    src/memory_store.cpp
    src/embedding.cpp
    src/injector.cpp
    src/sticky.cpp
    src/proxy.cpp
)

target_include_directories(memory-layer PRIVATE
    include
    deps
    deps/llama.cpp/include
    deps/llama.cpp/common
)

target_link_libraries(memory-layer PRIVATE
    llama
    sqlite3
)

# macOS frameworks for Metal + SIMD (Accelerate)
if(APPLE)
    target_link_libraries(memory-layer PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
        "-framework Accelerate"
    )
endif()

# --- Tests ---
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# --- Benchmark tool ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/bench_embedding.cpp")
    add_executable(bench_embedding
        tools/bench_embedding.cpp
        src/embedding.cpp
    )
    target_include_directories(bench_embedding PRIVATE
        include
        deps
        deps/llama.cpp/include
        deps/llama.cpp/common
    )
    target_link_libraries(bench_embedding PRIVATE llama sqlite3)
    if(APPLE)
        target_link_libraries(bench_embedding PRIVATE
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalKit"
            "-framework Accelerate"
        )
    endif()
endif()
