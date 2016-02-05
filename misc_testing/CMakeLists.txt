find_package(Boost COMPONENTS serialization filesystem iostreams system REQUIRED)
find_package(BZip2 REQUIRED)

add_executable(btree_test btree_test.cpp)
add_executable(btree_v2_test btree_v2_test.cpp)
add_executable(tokenize_test tokenize_tests.cpp)
add_executable(trie_test trie_test.cpp)
add_executable(trie_v2_test trie_v2_test.cpp)
add_executable(btree_drawer btree_drawer.cpp)

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
                      
add_executable(gpu_trie_test_v2 gpu_trie_test_v2.cpp)
target_link_libraries(gpu_trie_test_v2
                      memory_management
                      gpu_search_v2
                      )
                      
add_executable(interactive_gpu_test interactive_gpu_test.cpp)
target_link_libraries(interactive_gpu_test
                      memory_management
                      gpu_search
                      )
                      
add_executable(interactive_gpu_test_v2 interactive_gpu_test_v2.cpp)
target_link_libraries(interactive_gpu_test_v2
                      memory_management
                      gpu_search_v2
                      )

add_executable(gpu_LM_test gpu_LM_test.cpp)
target_link_libraries(gpu_LM_test
                      memory_management
                      gpu_search
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_IOSTREAMS_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${BZIP2_LIBRARIES}
                      )

add_executable(gpu_LM_test_v2 gpu_LM_test_v2.cpp)
target_link_libraries(gpu_LM_test_v2
                      memory_management
                      gpu_search_v2
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_IOSTREAMS_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${BZIP2_LIBRARIES}
                      )