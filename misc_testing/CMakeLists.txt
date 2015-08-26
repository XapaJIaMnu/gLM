add_executable(btree_test btree_test.cpp)
add_executable(tokenize_test tokenize_tests.cpp)
add_executable(trie_test trie_test.cpp)

add_executable(cuda_test cuda_test.cpp)
target_link_libraries(cuda_test
                      memory_management
                      gpu_search
                      )

add_executable(gpu_trie_test gpu_trie_test.cpp)
target_link_libraries(gpu_trie_test
                      memory_management
                      gpu_search
                      )