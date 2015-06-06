cmake_minimum_required (VERSION 3.0)
project (gLM)

#Debug build by default!
if (DEFINED ${BUILDTYPE})
    SET (CMAKE_CXX_FLAGS                "-O3 -Wall -std=c++11 -DNDEBUG")
    message("Building a release build with -O3.")
else()
    SET (CMAKE_CXX_FLAGS                "-g -Wall -std=c++11")
    SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
    message("Building a debug build.")
endif()

#Add include directories
include_directories ("${PROJECT_SOURCE_DIR}/Btree")
include_directories ("${PROJECT_SOURCE_DIR}/Parser")
include_directories ("${PROJECT_SOURCE_DIR}/misc")

add_subdirectory (Test)
add_subdirectory (misc_testing)
#add_subdirectory (Btree)
#add_subdirectory (Parser)
enable_testing ()
add_test (NAME AllTest COMMAND tests)

