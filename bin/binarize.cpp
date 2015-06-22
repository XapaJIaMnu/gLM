#include "serialization.hh"
#include "trie.hh"

int main(int argc, char* argv[]){
    if (argc != 4 && argc != 3) {
        std::cerr << "Usage:" << std::endl << argv[0] << " path_to_arpa_file output_path [btree_node_size]." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    unsigned short btree_node_size;

    if (argc == 3) {
        btree_node_size = 256;
    } else {
        btree_node_size = atoi(argv[3]);
    }
    //Create the LM
    LM lm;
    createTrieArray(argv[1], btree_node_size, lm);
    writeBinary(argv[2], lm);
    return 0;
}
