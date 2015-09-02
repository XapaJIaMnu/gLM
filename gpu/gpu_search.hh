//#include <cuda_runtime.h>
#include <stdio.h>

//__global__ void gpuSearchBtree(unsigned char * global_mem, unsigned int start_idx, int size, unsigned int key);

//Wrapper to call on the gpu
void searchWrapper(unsigned char * global_mem, unsigned int * keys, unsigned int num_keys, float * results);

void cudaDevSync();
