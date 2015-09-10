#include "trie.hh"
#include "lm_impl.hh"
#include "gpu_search.hh"

int main(int argc, char* argv[]){
    if (argc != 4 && argc != 3) {
        std::cerr << "Usage:" << std::endl << argv[0] << " path_to_arpa_file output_path [btree_node_size]." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    unsigned short btree_node_size;

    if (argc == 3) {
        btree_node_size = ENTRIES_PER_NODE;
    } else {
        btree_node_size = atoi(argv[3]);
    }
    //Create the LM
    LM lm;
    createTrieArray(argv[1], btree_node_size, lm);
    lm.writeBinary(argv[2], true);
    return 0;
}
