#include "search_utilities_v2.hh"

int main(int argc, char* argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " path_to_arpafile path_to_binary_model" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	testGPUsearch(argv[1], argv[2]);
}
