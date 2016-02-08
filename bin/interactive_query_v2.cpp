#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

int main(int argc, char* argv[]) {
    if (argc > 3 || argc == 0) {
        std::cerr << "Usage: " << argv[0] << " path_to_model_dir [gpuDeviceID=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 3) {
        setGPUDevice(atoi(argv[2]));
    }
    //Create the models
    LM lm(argv[1]);
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    interactiveRead(lm, btree_trie_gpu, first_lvl_gpu);

    //Free GPU memory
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);
    return 0;
}
