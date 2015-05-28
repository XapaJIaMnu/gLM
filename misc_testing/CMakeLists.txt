add_executable(btree_test btree_test.cpp)
add_executable(tokenize_test tokenize_tests.cpp)
#[[find_package(CUDA REQUIRED) # For when we add CUDA files
set(CUDA_NVCC_FLAGS "-arch=compute_30; -code=sm_30; -lcudart; -std=c++11")
cuda_add_executable(cuda_memcpytest cuda_memcpytest.cu)]]