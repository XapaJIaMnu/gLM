#include "gpu_LM_utils_v2.hh"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " path_to_arpafile entries_per_node" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    //Create the models
    LM lm;
    createTrie(argv[1],lm, atoi(argv[2])); //@Todo make this somewhat not so hardcoded
    std::pair<bool, std::string> cpu_test = test_trie(lm, argv[1], atoi(argv[2]));
    if (!cpu_test.first) {
        std::cout << "ERROR in CPU test: " << cpu_test.second;
    }
    std::cout << "Finished create Btree trie." << std::endl;
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, argv[1]);
    if (!res.first) {
        std::cerr << res.second << std::endl;
    }
    std::vector<unsigned int> ngrams = {5,2};
    //Entry_with_offset CPUres = searchTrie(lm.trieByteArray, lm.first_lvl, ngrams, atoi(argv[2]), false);
    //std::cout << "Prob: " << CPUres.prob << " Backoff: " << CPUres.backoff << std::endl;
    //std::cout << "Finished sanity check." << std::endl;
    interactiveRead(lm, btree_trie_gpu, first_lvl_gpu);

    //Free GPU memory
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);
    return 0;
}
