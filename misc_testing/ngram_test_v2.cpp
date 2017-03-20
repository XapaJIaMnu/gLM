#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc == 0) {
        std::cerr << "Usage: " << argv[0] << " path_to_model_dir vocabID1 vocabID2 ... vocabIDN [gpuDeviceID=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    //Create the models
    LM lm(argv[1]);
    int num_args = 2 + lm.metadata.max_ngram_order;

    int gpuDeviceID = 0;
    if (argc == num_args + 1) {
        gpuDeviceID = atoi(argv[num_args]);
    }

    GPUSearcher engine(1, lm, gpuDeviceID);

    unsigned int * ngram = new unsigned int[lm.metadata.max_ngram_order];

    for (int i = 0; i<lm.metadata.max_ngram_order; i++) {
        ngram[i] = atoi(argv[2+i]);
    }
    
    unsigned int * device_IDs = copyToGPUMemory(ngram, lm.metadata.max_ngram_order);
    float * gpuResults;
    allocateGPUMem(1, &gpuResults);

    engine.search(device_IDs, 1, gpuResults, 0);

    float * hostResults = new float[1];
    copyToHostMemory(gpuResults, hostResults, 1);

    std::cout << "Query successful, score: " << hostResults[0] << std::endl;

    return 0;
}
