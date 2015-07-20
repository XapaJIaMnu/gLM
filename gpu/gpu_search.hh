//#include <cuda_runtime.h>
#include <stdio.h>

//__global__ void gpuSearchBtree(unsigned char * global_mem, unsigned int start_idx, int size, unsigned int key);

//Wrapper to call on the gpu
void searchWrapper(unsigned char * global_mem, unsigned int start_idx, int first_size, unsigned int key, int grid, int block);

void cudaDevSync();
