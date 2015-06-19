#include "trie.hh"

int main(int argc, char* argv[]) {
    std::vector<unsigned char> byte_arr;
    LM_metadata metadata = createTrieArray(argv[1], atoi(argv[2]), byte_arr);
    std::cout << metadata;
} 
