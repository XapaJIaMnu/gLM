#include <stdio.h>

//Wrapper to call on the gpu
void searchWrapper(unsigned char * global_mem, unsigned int * keys, unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int entry_size, unsigned int max_ngram);

void cudaDevSync();
