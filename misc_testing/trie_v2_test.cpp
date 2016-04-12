#include "trie_v2.hh"

int main(int argc, char* argv[]) {
    //LM lm;
    //createTrieArray(argv[1], atoi(argv[2]), lm);
    std::pair<bool, std::string> res = test_trie(argv[1], atoi(argv[2]));
    //std::cout << lm.metadata;
    std::cout << "True: " << res.first << std::endl << res.second << std::endl;
} 
