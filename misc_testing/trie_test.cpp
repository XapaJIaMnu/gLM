#include "trie.hh"

int main(int argc, char* argv[]) {
    ArpaReader pesho(argv[1]);
    processed_line text = pesho.readline();
    B_tree * root_btree = new B_tree(atoi(argv[2]));

    while (!text.filefinished){
        addToTrie(root_btree, text, pesho.max_ngrams, atoi(argv[2]));
        text = pesho.readline();
    }
    std::cout << "Finished constructing the trie!" << std::endl;

    std::cout << "Total trie array size is: " << calculateTrieSize(root_btree)/(1024*1024) << "MB." << std::endl;
} 
