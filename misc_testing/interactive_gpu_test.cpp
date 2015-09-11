#include "gpu_LM_utils.hh"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " path_to_arpafile entries_per_node" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    //Create the models
    LM lm;
    createTrieArray(argv[1], atoi(argv[2]), lm); //@Todo make this somewhat not so hardcoded
    std::cout << "Finished create Btree trie." << std::endl;
    unsigned char * gpuByteArray = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());

    std::pair<bool, std::string> res = testQueryNgrams(lm, gpuByteArray, argv[1]);
    if (!res.first) {
        std::cerr << res.second << std::endl;
    }
    std::cout << "Finished sanity check." << std::endl;
    interactiveRead(lm, gpuByteArray);

    //Free GPU memory
    freeGPUMemory(gpuByteArray);
    return 0;
}
