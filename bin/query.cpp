#include "serialization.hh"

int main(int argc, char* argv[]){
	if (argc != 2) {
		std::cerr << "Usage:" << std::endl << argv[0] << " path_to_binary_lm_dir" << std::endl;
	}
	std::vector<unsigned char> byte_arr;
	LM_metadata metatada = readBinary(argv[1], byte_arr);
    std::cout << "STUB! Read a binary LM with parameters:" << std::endl << metatada << "Nothing more to do.";
    return 0;
}
