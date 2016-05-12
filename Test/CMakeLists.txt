find_package(Boost COMPONENTS unit_test_framework filesystem system REQUIRED)

#Get the filepath of the test file and define it.
get_filename_component(arpaFilePath "../arpa/toy_lm.arpa" REALPATH [CACHE])
add_definitions(-DARPA_TESTFILEPATH="${arpaFilePath}")
add_executable(tests tests.cpp)

target_link_libraries(tests
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                    )
                    
add_executable(lm_tests lm_tests.cpp)
target_link_libraries(lm_tests
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                    )
                    
add_executable(btree_tests btree_tests.cpp)
target_link_libraries(btree_tests
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                    )

add_executable(gpu_tests_suite gpu_test_suite.cpp)
target_link_libraries(gpu_tests_suite
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      memory_management
                      gpu_search
                     )
                     
add_executable(gpu_tests_suite_v2 gpu_test_suite_v2.cpp)
target_link_libraries(gpu_tests_suite_v2
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      memory_management
                      gpu_search_v2
                     )
