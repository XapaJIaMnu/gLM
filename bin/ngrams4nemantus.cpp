#include "nematus_ngrams.hh"

int main(int argc, char* argv[]) {
    if (!(argc != 5 || argc != 6)) {
        std::cerr << "Usage: " << argv[0] << " path_to_model_dir path_to_ngrams_file path_to_vocab_file maxGPUMemoryMB [gpuDeviceID=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    //Results vector:
    int gpuDeviceID = 0;
    if (argc == 6) {
        gpuDeviceID = atoi(argv[5]);
    }

    NematusLM ngramEngine = NematusLM(argv[1], argv[3], std::stoull(argv[4]), gpuDeviceID);
    float * results = ngramEngine.processBatch(argv[2]);
    results[0] = results[0]; //Silence compiler warning.

    ngramEngine.freeResultsMemory();

    return 0;
}
