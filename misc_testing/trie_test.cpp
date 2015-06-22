#include "trie.hh"

int main(int argc, char* argv[]) {
    LM lm;
    createTrieArray(argv[1], atoi(argv[2]), lm);
    std::cout << lm.metadata;
} 
