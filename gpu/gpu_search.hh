//#include <cuda_runtime.h>
#include <stdio.h>
#define MAX_NUM_CHILDREN 128
#define ENTRIES_PER_NODE (MAX_NUM_CHILDREN - 1)
#define ENTRY_SIZE (sizeof(unsigned int) + sizeof(unsigned int) + 2*sizeof(float)) //Same as the getEntrySize(true)
#define MAX_NGRAM 5
//Assume working with 128 thread DON'T RELY ENTIRERLY ON THOSE! Size may be smaller. need a parameter.

//__global__ void gpuSearchBtree(unsigned char * global_mem, unsigned int start_idx, int size, unsigned int key);

//Wrapper to call on the gpu
void searchWrapper(unsigned char * global_mem, unsigned int * keys, unsigned int num_ngram_queries, float * results);

void cudaDevSync();
