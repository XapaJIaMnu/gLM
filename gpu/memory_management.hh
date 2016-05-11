#include <stddef.h>

unsigned char * copyToGPUMemory(unsigned char * byte_arr, size_t byte_arr_size);
void freeGPUMemory(unsigned char * gpu_ptr);

unsigned int * copyToGPUMemory(unsigned int * byte_arr, size_t num_elements);

void copyToGPUMemoryNoAlloc(unsigned int * gpuMem, unsigned int * input, size_t num_elements);

void copyToGPUMemoryNoAlloc(float * gpuMem, float * input, size_t num_elements);


void freeGPUMemory(unsigned int * gpu_ptr);

void freeGPUMemory(float * gpu_ptr);

void allocateGPUMem(size_t num_elements, unsigned int ** gpu_mem);
void allocateGPUMem(size_t num_elements, float ** gpu_mem);
void copyToHostMemory(float * gpu_mem, float * cpu_mem, size_t num_elements);
void copyToHostMemory(unsigned int * gpu_mem, unsigned int * cpu_mem, size_t num_elements);
