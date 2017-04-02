if (${DEBUGBUILD} EQUAL 0)
    set(CUDA_NVCC_FLAGS "-lcudart; -O3")
else()
    set(CUDA_NVCC_FLAGS "-lcudart; -g; -G")
    message("Building a debug gpu build.")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -code=sm_${GPU_COMPUTE_VER}; -arch=compute_${GPU_COMPUTE_VER}")

#Workaround for gcc 6.0+ and cuda < 8
if(CUDA_VERSION_MAJOR VERSION_LESS 8)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11; -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO")
endif()

#Workaround on old cuda versions withbad host flags
if((CUDA_VERSION VERSION_LESS 7.5) OR (${BAD_HOST_FLAGS} EQUAL 1))
    message("Doing some bad things because our cuda version is too old or our host compiler has weird flags.")
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES")
endif()

cuda_add_library(memory_management memory_management.cu)
cuda_add_library(gpu_search gpu_search.cu)
cuda_add_library(gpu_search_v2 gpu_search_v2.cu)
target_link_libraries(gpu_search_v2
                      memory_management)

cuda_add_library(memory_management_FPIC SHARED memory_management.cu)
cuda_add_library(gpu_search_FPIC SHARED gpu_search.cu)
cuda_add_library(gpu_search_v2_FPIC SHARED gpu_search_v2.cu)
target_link_libraries(gpu_search_v2_FPIC
                      memory_management_FPIC)
