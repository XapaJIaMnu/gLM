#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include "lm.hh"

//Wrapper to call on the gpu
void searchWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys, unsigned int num_ngram_queries,
 float * results, unsigned int entries_per_node, unsigned int max_ngram, bool make_exp = false, bool debug = true);
void searchWrapperStream(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram, cudaStream_t& stream, bool make_exp = false, bool debug = false);

void cudaDevSync();

/*Tells the code to execute on a particular device. Useful on multiGPU systems*/
void setGPUDevice(int deviceID);

class GPUSearcher {
    private:
        cudaStream_t * streams;
        int num_streams;
        bool make_exp;
        
        //GPU pointers
        unsigned char * btree_trie_gpu;
        unsigned int * first_lvl_gpu;
        void gpuInit();
        
    public:
        LM& lm;
        void search(unsigned int * keys, unsigned int num_ngram_queries, float * results, int streamID, bool debug = false);
        GPUSearcher(int, LM&, bool = false);
        GPUSearcher(int, LM&, int, bool = false);
        ~GPUSearcher();
};
