#include "tokenizer.hh"
#include "serialization.hh"
#include "memory_management.hh"
#include "gpu_search.hh"

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
        check_against.push_back(text2.backoff);

        num_keys++;
        text2 = pesho2.readline();
        if (num_keys > 500000) {
            break;
        }
    }

    total_num_keys = num_keys;
}

template<class StringType>
void testGPUsearch(StringType arpafile, StringType pathTobinary) {
    LM lm; //The read in language model
    readBinary(pathTobinary, lm);

    //unsigned int max_btree_node_size = lm.metadata.btree_node_size; Will use when we move away from hardcoding it
    unsigned short max_ngram_order = lm.metadata.max_ngram_order;

    //Prepare for search

    std::vector<unsigned int> keys;
    std::vector<float> check_against;

    std::cout << "Read in binary, preparing search vectors..." << std::endl;
    unsigned int total_num_keys;
    prepareSearchVectors(keys, check_against, max_ngram_order, total_num_keys, arpafile);

    //Search every single key
    unsigned char * gpuByteArray = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());

    unsigned int * gpuKeys = copyToGPUMemory(keys.data(), keys.size());
    unsigned int * results;
    allocateGPUMem(total_num_keys*3, &results);

    searchWrapper(gpuByteArray, gpuKeys, total_num_keys, results);

    //Copy back to host
    unsigned int * results_cpu = new unsigned int[total_num_keys*3];
    copyToHostMemory(results, results_cpu, total_num_keys*3);

    //Clear gpu memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(gpuByteArray);
    freeGPUMemory(results);

    //Compare results

    for (unsigned int i = 0; i < total_num_keys; i++) {
        float res_prob = *(float *)&results_cpu[i*3 + 1];
        float res_backoff = *(float *)&results_cpu[i*3 + 2];

        float exp_prob = check_against[i*2];
        float exp_backoff = check_against[i*2 + 1];

        if (!((exp_prob == res_prob) && (exp_backoff == res_backoff))) {
            std::cout << "Error expected prob: " << exp_prob << " and backoff: " << exp_backoff << " got: "
                << res_prob << " and " << res_backoff << std::endl;
        }
    }
    delete[] results_cpu;
}

