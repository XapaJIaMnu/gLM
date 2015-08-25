#include "gpu_common.h"
#include "memory_management.hh"

unsigned char * copyToGPUMemory(unsigned char * byte_arr, unsigned int byte_arr_size) {
    unsigned char * gpu_byte_arr;
    CHECK_CALL(cudaMalloc(&gpu_byte_arr, byte_arr_size*sizeof(unsigned char)));
    CHECK_CALL(cudaMemcpy(gpu_byte_arr, byte_arr, byte_arr_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    return gpu_byte_arr;
}

unsigned int * copyToGPUMemory(unsigned int * byte_arr, unsigned int byte_arr_size) {
    unsigned int * gpu_byte_arr;
    CHECK_CALL(cudaMalloc(&gpu_byte_arr, byte_arr_size*sizeof(unsigned int)));
    CHECK_CALL(cudaMemcpy(gpu_byte_arr, byte_arr, byte_arr_size*sizeof(unsigned int), cudaMemcpyHostToDevice));
    return gpu_byte_arr;
}

void allocateGPUMem(size_t size, unsigned int ** gpu_mem) {
    CHECK_CALL(cudaMalloc(gpu_mem, size*sizeof(unsigned int)));
}

void copyToHostMemory(unsigned int * gpu_mem, unsigned int * cpu_mem, size_t size) {
    CHECK_CALL(cudaMemcpy(cpu_mem, gpu_mem, size*sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void freeGPUMemory(unsigned char * gpu_ptr) {
    CHECK_CALL(cudaFree(gpu_ptr));
}

void freeGPUMemory(unsigned int * gpu_ptr) {
    CHECK_CALL(cudaFree(gpu_ptr));
}
