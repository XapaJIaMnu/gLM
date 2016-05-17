find_package(CUDA REQUIRED) # For when we add CUDA files
if (${DEBUGBUILD} EQUAL 0)
    set(CUDA_NVCC_FLAGS "-arch=compute_52; -code=sm_52; -lcudart; -O3")
else()
    set(CUDA_NVCC_FLAGS "-arch=compute_52; -code=sm_52; -lcudart; -g; -G")
    message("Building a debug gpu build.")
endif()

#Workaround for gcc 6.0+ and cuda < 8
if((NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0) AND (CUDA_VERSION_MAJOR VERSION_LESS 8))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11; -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO")
endif()

cuda_add_library(memory_management memory_management.cu)
cuda_add_library(gpu_search gpu_search.cu)
cuda_add_library(gpu_search_v2 gpu_search_v2.cu)
