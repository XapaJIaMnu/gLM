#include "gpu_common.h"
#include "memory_management.hh"

unsigned char * copyToGPUMemory(unsigned char * byte_arr, unsigned int byte_arr_size) {
    unsigned char * gpu_byte_arr;
    CHECK_CALL(cudaMalloc(&gpu_byte_arr, byte_arr_size*sizeof(unsigned char)));
    CHECK_CALL(cudaMemcpy(gpu_byte_arr, byte_arr, byte_arr_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    return gpu_byte_arr;
}

void freeGPUMemory(unsigned char * gpu_ptr) {
    CHECK_CALL(cudaFree(gpu_ptr));
}
