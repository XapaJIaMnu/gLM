#include "trie.hh"

int main(int argc, char* argv[]) {
    B_tree * root_btree = createTrie(argv[1], atoi(argv[2]));
    std::cout << "Finished constructing the trie!" << std::endl;

    size_t trie_size = calculateTrieSize(root_btree);
    std::cout << "Potential serialized size is: " << trie_size/(1024*1024) << " MB."<< std::endl;
    //Convert it to array
    std::vector<unsigned char>byte_arr;
    std::cout << "Reserved size is: " << trie_size << " bytes." << std::endl;
    byte_arr.reserve(trie_size);
    trieToByteArray(byte_arr, root_btree);
    deleteTrie(root_btree);
} 
