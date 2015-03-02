cmake_minimum_required (VERSION 3.0)
project (gLM)

#Work with clang for now later add a switch to gcc
SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS                "-g -Wall -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_AR      "/usr/bin/llvm-ar")
SET (CMAKE_LINKER  "/usr/bin/llvm-ld")
SET (CMAKE_NM      "/usr/bin/llvm-nm")
SET (CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
SET (CMAKE_RANLIB  "/usr/bin/llvm-ranlib")

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

