cmake_minimum_required (VERSION 3.0)
project (gLM C CXX)

set(DEBUGBUILD 0)
set(BAD_HOST_FLAGS 0)
set(GPU_COMPUTE_VER 52)

#Fix GPU compilation on bad ubuntu systems
if (DEFINED BAD_HOST)
    if ((${BAD_HOST} EQUAL 1) OR (${BAD_HOST} EQUAL 0))
        set(BAD_HOST_FLAGS ${BAD_HOST})
    else()
        message(SEND_ERROR "Invalid value for the BAD_HOST variable: ${BAD_HOST}. Valid values are 1 (for true) and 0 (for false)")
    endif()
endif()

if (DEFINED COMPUTE_VER)
    message("Setting custom compute version target for the GPU compilation: ${COMPUTE_VER}. If the version specified is invalid compilation will fail. If the version specified is higher than what your GPU supports, it will compile successfully but the tests won't pass.")
    set(GPU_COMPUTE_VER ${COMPUTE_VER})
endif()

#Release build by default
if (DEFINED BUILDTYPE)
    if (${BUILDTYPE} STREQUAL "release")
        message("Building a release build with -O3.")
    elseif(${BUILDTYPE} STREQUAL "debug")
        set(DEBUGBUILD 1)
        message("Building a debug build.")
    else()
        message(SEND_ERROR "Unrecognized build type. Valid values are \"release\" or \"debug\". Building a release build.")
    endif()
else()
        message("Building a release build with -O3.")
endif()

if(${DEBUGBUILD} EQUAL 1)
    SET (CMAKE_CXX_FLAGS                "-g -Wall")
    SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
else()
    SET (CMAKE_CXX_FLAGS                "-O3 -Wall -DNDEBUG")
endif()

#GCC 6.0+ defaults to c++14 so no need to specify c++11 manually
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

#Add include directories
include_directories ("${PROJECT_SOURCE_DIR}/Btree")
include_directories ("${PROJECT_SOURCE_DIR}/Trie")
include_directories ("${PROJECT_SOURCE_DIR}/Parser")
include_directories ("${PROJECT_SOURCE_DIR}/misc")
include_directories ("${PROJECT_SOURCE_DIR}/gpu")
include_directories ("${PROJECT_SOURCE_DIR}/LM")

add_subdirectory (Test)
add_subdirectory (misc_testing)
add_subdirectory (bin)
add_subdirectory (gpu)
add_subdirectory (LM)
#add_subdirectory (Btree)
#add_subdirectory (Parser)
enable_testing ()
add_test (NAME AllTest COMMAND tests)
add_test (NAME gpu_test COMMAND gpu_tests_suite)
add_test (NAME gpu_test_v2 COMMAND gpu_tests_suite_v2)
add_test (NAME btree_test COMMAND btree_tests)
add_test (NAME lm_test COMMAND lm_tests)

