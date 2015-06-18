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
    //Compress the trie
    compressTrie(root_btree);

    size_t trie_size = calculateTrieSize(root_btree);
    std::cout << "Potential serialized size is: " << trie_size/(1024*1024) << " MB."<< std::endl;
    //Convert it to array
    std::vector<unsigned char>byte_arr;
    std::cout << "Reserved size is: " << trie_size << " bytes." << std::endl;
    byte_arr.reserve(trie_size);
    trieToByteArray(byte_arr, root_btree);
    std::cout << "Size of byte_arr: " << byte_arr.size() << std::endl;

    deleteTrie(root_btree);

    std::pair<bool, std::string> test = test_trie(argv[1], atoi(argv[2]));

    std::cout << "Correct: " << test.first << " res: " << test.second << std::endl;
} 
