#include <cuda_runtime.h>

__device__ void copyToSharedMemory(unsigned char * global_mem, int start_idx, int size, unsigned int key);
