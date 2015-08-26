#include "trie.hh"
#include "memory_management.hh"
#include "gpu_search.hh"
#include <chrono>
#include <ctime>

int main(int argc, char* argv[]) {
    LM lm;
    createTrieArray(argv[1], atoi(argv[2]), lm);

    unsigned short max_ngram_order = lm.metadata.max_ngram_order;

    ArpaReader pesho2(argv[1]);
    processed_line text2 = pesho2.readline();

    unsigned char * gpuByteArray = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    std::cout << "Byte array size: " << lm.trieByteArray.size() << std::endl;

    std::vector<unsigned int> keys_to_query;
    unsigned int num_keys = 0; //How many ngrams are we going to query

    while (!text2.filefinished) {
        //Inefficient reallocation of keys_to_query. Should be done better
        for (int i = 0; i < (max_ngram_order - text2.ngrams.size()); i++) {
            text2.ngrams.push_back(0); //Extend ngrams to max num ngrams if they are of lower order
        }
        //keys_to_query.resize(keys_to_query.size() + text2.ngrams.size()); //@TODO redo when not sleep deprived
        //std::memcpy(keys_to_query.data() + keys_to_query.size() - text2.ngrams.size(), text2.ngrams.data(), text2.ngrams.size());
        //std::cout << "Max ngram: " << max_ngram_order << std::endl;
        for (unsigned int i = 0; i < max_ngram_order; i++) {
            keys_to_query.push_back(text2.ngrams[i]); //Extend ngrams to max num ngrams if they are of lower order
        }

        num_keys++;
        text2 = pesho2.readline();
    }

    unsigned int * gpuKeys = copyToGPUMemory(keys_to_query.data(), keys_to_query.size());
    unsigned int * results;
    allocateGPUMem(num_keys*3, &results);
    searchWrapper(gpuByteArray, gpuKeys, num_keys, results);

    //Copy back to host
    unsigned int * results_cpu = new unsigned int[num_keys*3];
    copyToHostMemory(results, results_cpu, num_keys*3);

    freeGPUMemory(gpuByteArray);
    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    delete[] results_cpu;
}