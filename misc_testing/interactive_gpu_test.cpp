#include "gpu_tests.hh"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " path_to_arpafile." << std::endl;
    }
    std::pair<bool, std::string> res = testQueryNgrams(argv[1]);
    if (!res.first) {
        std::cerr << res.second << std::endl;
    }
    return 0;
}
