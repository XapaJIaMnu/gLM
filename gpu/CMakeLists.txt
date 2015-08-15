find_package(CUDA REQUIRED) # For when we add CUDA files
set(CUDA_NVCC_FLAGS "-arch=compute_52; -code=sm_52; -lcudart")
cuda_add_library(memory_management memory_management.cu)
cuda_add_library(gpu_search gpu_search.cu)
