#pragma once
//#include "tokenizer.hh"
#include "memory_management.hh"
#include "gpu_search_v2.hh"
#include "trie_v2_impl.hh"
#include "lm_impl.hh"

template<class StringType>
void prepareSearchVectors(std::vector<unsigned int>& keys, std::vector<float>& check_against, unsigned short max_ngram_order, unsigned int& total_num_keys, StringType arpafile) {
    ArpaReader pesho2(arpafile);
    processed_line text2 = pesho2.readline();

    unsigned int num_keys = 0; //How many ngrams are we going to query

    while (!text2.filefinished) {
        //Inefficient reallocation of keys_to_query. Should be done better
        int num_padded =  max_ngram_order - text2.ngrams.size();
        for (int i = 0; i < num_padded; i++) {
            text2.ngrams.push_back(0); //Extend ngrams to max num ngrams if they are of lower order
        }
        
        for (unsigned int i = 0; i < max_ngram_order; i++) {
            keys.push_back(text2.ngrams[i]); //Extend ngrams to max num ngrams if they are of lower order
        }

        check_against.push_back(text2.score);

        num_keys++;
        text2 = pesho2.readline();
    }

    total_num_keys = num_keys;
}

template<class StringType>
void testGPUsearch(StringType arpafile, StringType pathTobinary) {
    LM lm(pathTobinary);

    //unsigned int max_btree_node_size = lm.metadata.btree_node_size; Will use when we move away from hardcoding it
    unsigned short max_ngram_order = lm.metadata.max_ngram_order;

    std::vector<unsigned int> keys;
    std::vector<float> check_against;

    std::cout << "Read in binary, preparing search vectors..." << std::endl;
    unsigned int total_num_keys;
    prepareSearchVectors(keys, check_against, max_ngram_order, total_num_keys, arpafile);

    //Search every single key
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    unsigned int * gpuKeys = copyToGPUMemory(keys.data(), keys.size());
    float * results;
    allocateGPUMem(total_num_keys, &results);

    searchWrapper(btree_trie_gpu, first_lvl_gpu, gpuKeys, total_num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    float * results_cpu = new float[total_num_keys];
    copyToHostMemory(results, results_cpu, total_num_keys);

    //Clear gpu memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(results);
    freeGPUMemory(first_lvl_gpu);

    //Compare results

    for (unsigned int i = 0; i < total_num_keys; i++) {
        float res_prob = *(float *)&results_cpu[i];


        float exp_prob = check_against[i];

        if (!(exp_prob == res_prob)) {
            std::cout << "Error expected prob: " << exp_prob << " got: "
                << res_prob << " ." << std::endl;
        }
    }
    delete[] results_cpu;
}

