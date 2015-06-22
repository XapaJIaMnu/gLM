#include "serialization.hh"

int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "Usage:" << std::endl << argv[0] << " path_to_binary_lm_dir" << std::endl;
    }
    LM lm; //The read in language model
    readBinary(argv[1], lm);
    std::cout << "STUB! Read a binary LM with parameters:" << std::endl << lm.metadata << "Nothing more to do.";
    return 0;
}
