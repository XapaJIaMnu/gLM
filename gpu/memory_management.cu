#include "gpu_common.h"
#include "memory_management.hh"

unsigned char * copyToGPUMemory(unsigned char * byte_arr, size_t byte_arr_size) {
    unsigned char * gpu_byte_arr;
    CHECK_CALL(cudaMalloc(&gpu_byte_arr, byte_arr_size*sizeof(unsigned char)));
    CHECK_CALL(cudaMemcpy(gpu_byte_arr, byte_arr, byte_arr_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    return gpu_byte_arr;
}

unsigned int * copyToGPUMemory(unsigned int * byte_arr, size_t num_elements) {
    unsigned int * gpu_byte_arr;
    CHECK_CALL(cudaMalloc(&gpu_byte_arr, num_elements*sizeof(unsigned int)));
    CHECK_CALL(cudaMemcpy(gpu_byte_arr, byte_arr, num_elements*sizeof(unsigned int), cudaMemcpyHostToDevice));
    return gpu_byte_arr;
}

void copyToGPUMemoryNoAlloc(unsigned int * gpuMem, unsigned int * input, size_t num_elements) {

    CHECK_CALL(cudaMemcpy(gpuMem, input, num_elements*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void copyToGPUMemoryNoAlloc(float * gpuMem, float * input, size_t num_elements) {

    CHECK_CALL(cudaMemcpy(gpuMem, input, num_elements*sizeof(float), cudaMemcpyHostToDevice));
}


void allocateGPUMem(size_t num_elements, unsigned int ** gpu_mem) {
    CHECK_CALL(cudaMalloc(gpu_mem, num_elements*sizeof(unsigned int)));
}

void allocateGPUMem(size_t num_elements, float ** gpu_mem) {
    CHECK_CALL(cudaMalloc(gpu_mem, num_elements*sizeof(float)));
}

void copyToHostMemory(float * gpu_mem, float * cpu_mem, size_t num_elements) {
    CHECK_CALL(cudaMemcpy(cpu_mem, gpu_mem, num_elements*sizeof(float), cudaMemcpyDeviceToHost));
}

void freeGPUMemory(unsigned char * gpu_ptr) {
    CHECK_CALL(cudaFree(gpu_ptr));
}

void freeGPUMemory(float * gpu_ptr) {
    CHECK_CALL(cudaFree(gpu_ptr));
}

void freeGPUMemory(unsigned int * gpu_ptr) {
    CHECK_CALL(cudaFree(gpu_ptr));
}

void pinnedMemoryAllocator(unsigned int * pinned_mem, size_t num_elements) {
    CHECK_CALL(cudaHostAlloc(&pinned_mem, num_elements*sizeof(unsigned int), cudaHostAllocDefault));
}
void pinnedMemoryAllocator(float * pinned_mem, size_t num_elements) {
    CHECK_CALL(cudaHostAlloc(&pinned_mem, num_elements*sizeof(float), cudaHostAllocDefault));
}
