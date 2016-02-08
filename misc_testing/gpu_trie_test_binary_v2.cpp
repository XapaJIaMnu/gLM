#include "lm_impl.hh"
#include "memory_management.hh"
#include "gpu_search_v2.hh"
#include <sstream>
#include <chrono>
#include <ctime>
#include <boost/tokenizer.hpp>

int main(int argc, char* argv[]) {
    LM lm(argv[1]);

    //Initiate gpu search
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //input ngram
    std::string response;
    boost::char_separator<char> sep(" ");
    while (true) {
        getline(std::cin, response);
        if (response == "/end") {
            break;
        }

        std::vector<unsigned int> keys_to_query;
        boost::tokenizer<boost::char_separator<char> > tokens(response, sep);
        for (auto word : tokens) {
            keys_to_query.push_back(std::stoul(word, 0, 10));
        }

        unsigned int num_keys = 1; //How many ngrams are we going to query

        unsigned int * gpuKeys = copyToGPUMemory(keys_to_query.data(), keys_to_query.size());
        float * results;
        allocateGPUMem(num_keys, &results);

        searchWrapper(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

        //Copy back to host
        float * results_cpu = new float[num_keys];
        copyToHostMemory(results, results_cpu, num_keys);

        std::cout << "Query result is: " << results_cpu[0] << std::endl;

        for (auto key : keys_to_query) {
            std::cout << lm.decode_map.find(key)->second << " ";
        }
        std::cout << std::endl;

        freeGPUMemory(gpuKeys);
        freeGPUMemory(results);

        delete[] results_cpu;
    }

    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);
}
