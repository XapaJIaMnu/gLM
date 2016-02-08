#include "gpu_LM_utils.hh"
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
    unsigned char * gpuByteArray = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());

    interactiveRead(lm, gpuByteArray);
    //Free GPU memory
    freeGPUMemory(gpuByteArray);
    return 0;
}
