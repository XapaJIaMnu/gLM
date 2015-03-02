find_package(Boost COMPONENTS unit_test_framework REQUIRED)
add_executable(tests tests.cpp)

target_link_libraries(tests
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
                      )
                      
enable_testing()
add_test(AllTest tests)