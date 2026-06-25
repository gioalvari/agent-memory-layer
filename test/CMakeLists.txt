add_executable(test_memory_store test_memory_store.cpp ../src/memory_store.cpp)
target_include_directories(test_memory_store PRIVATE ../include ../deps)
target_link_libraries(test_memory_store PRIVATE sqlite3)
if(APPLE)
    target_link_libraries(test_memory_store PRIVATE "-framework Accelerate")
endif()
add_test(NAME test_memory_store COMMAND test_memory_store)

add_executable(test_injector test_injector.cpp ../src/injector.cpp)
target_include_directories(test_injector PRIVATE ../include ../deps)
add_test(NAME test_injector COMMAND test_injector)

add_executable(test_config test_config.cpp ../src/config.cpp)
target_include_directories(test_config PRIVATE ../include ../deps)
add_test(NAME test_config COMMAND test_config)

add_executable(test_cosine test_cosine.cpp ../src/memory_store.cpp)
target_include_directories(test_cosine PRIVATE ../include ../deps)
target_link_libraries(test_cosine PRIVATE sqlite3)
if(APPLE)
    target_link_libraries(test_cosine PRIVATE "-framework Accelerate")
endif()
add_test(NAME test_cosine COMMAND test_cosine)

add_executable(test_scoring test_scoring.cpp ../src/memory_store.cpp)
target_include_directories(test_scoring PRIVATE ../include ../deps)
target_link_libraries(test_scoring PRIVATE sqlite3)
if(APPLE)
    target_link_libraries(test_scoring PRIVATE "-framework Accelerate")
endif()
add_test(NAME test_scoring COMMAND test_scoring)

# Integration test: requires llama.cpp for EmbeddingWorker (even if model won't load)
add_executable(test_integration test_integration.cpp
    ../src/proxy.cpp
    ../src/memory_store.cpp
    ../src/embedding.cpp
    ../src/injector.cpp
    ../src/config.cpp
)
target_include_directories(test_integration PRIVATE
    ../include
    ../deps
    ../deps/llama.cpp/include
    ../deps/llama.cpp/common
)
target_link_libraries(test_integration PRIVATE llama sqlite3)
if(APPLE)
    target_link_libraries(test_integration PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
        "-framework Accelerate"
    )
endif()
add_test(NAME test_integration COMMAND test_integration)

# Benchmark: serial vs batch embedding throughput (NOT a ctest — run manually with a real model)
add_executable(bench_embedding bench_embedding.cpp ../src/embedding.cpp)
target_include_directories(bench_embedding PRIVATE
    ../include ../deps
    ../deps/llama.cpp/include ../deps/llama.cpp/common
)
target_link_libraries(bench_embedding PRIVATE llama)
if(APPLE)
    target_link_libraries(bench_embedding PRIVATE
        "-framework Foundation" "-framework Metal"
        "-framework MetalKit"   "-framework Accelerate"
    )
endif()
