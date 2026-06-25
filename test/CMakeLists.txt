add_executable(test_memory_store test_memory_store.cpp ../src/memory_store.cpp)
target_include_directories(test_memory_store PRIVATE ../include ../deps)
target_link_libraries(test_memory_store PRIVATE sqlite3)
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
add_test(NAME test_cosine COMMAND test_cosine)
