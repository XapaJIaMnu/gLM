find_package(Boost COMPONENTS unit_test_framework serialization filesystem iostreams system REQUIRED)

find_package(BZip2 REQUIRED)

#Get the filepath of the test file and define it.
get_filename_component(arpaFilePath "../arpa/toy_lm.arpa" REALPATH [CACHE])
add_definitions(-DARPA_TESTFILEPATH="${arpaFilePath}")
add_executable(tests tests.cpp)

target_link_libraries(tests
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${Boost_IOSTREAMS_LIBRARY}
                      ${BZIP2_LIBRARIES}
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