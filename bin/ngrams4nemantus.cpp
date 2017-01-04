#include "nematus_ngrams.hh"

int main(int argc, char* argv[]) {
    if (!(argc != 5 || argc != 6)) {
        std::cerr << "Usage: " << argv[0] << " path_to_model_dir path_to_ngrams_file path_to_vocab_file maxGPUMemoryMB [gpuDeviceID=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 6) {
        initModel(argv[1], argv[2], argv[3], std::stoull(argv[4]), atoi(argv[5]));
    } else {
        initModel(argv[1], argv[2], argv[3], std::stoull(argv[4]));
    }
    
    return 0;
}
