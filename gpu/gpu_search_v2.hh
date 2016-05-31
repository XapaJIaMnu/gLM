#pragma once
 #include <stdio.h>
struct cudaStream_t; //Forward declaration

//Wrapper to call on the gpu
void searchWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys, unsigned int num_ngram_queries,
 float * results, unsigned int entries_per_node, unsigned int max_ngram);
void searchWrapperStream(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram, cudaStream_t& stream);

void cudaDevSync();

/*Tells the code to execute on a particular device. Useful on multiGPU systems*/
void setGPUDevice(int deviceID);

class GPUSearcher {
    private:
        cudaStream_t * streams;
        int num_streams;
    public:
        //GPUSearcher(int);
        GPUSearcher();
        ~GPUSearcher();
};
