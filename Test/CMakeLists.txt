find_package(Boost COMPONENTS unit_test_framework serialization filesystem system REQUIRED)

#Get the filepath of the test file and define it.
get_filename_component(arpaFilePath "../arpa/toy_lm.arpa" REALPATH [CACHE])
add_definitions(-DARPA_TESTFILEPATH="${arpaFilePath}")
add_executable(tests tests.cpp)

target_link_libraries(tests
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
		     )
