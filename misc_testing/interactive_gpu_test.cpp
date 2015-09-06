#include "gpu_tests.hh"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " path_to_arpafile." << std::endl;
    }
    //Create the models
    LM lm;
    createTrieArray(argv[1], ENTRIES_PER_NODE, lm); //@Todo make this somewhat not so hardcoded
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
