#include "trie.hh"
#include "lm_impl.hh"
#include "gpu_search.hh"

int main(int argc, char* argv[]){
    if (argc != 5 && argc != 3) {
        std::cerr << "Usage:" << std::endl << argv[0] << " path_to_arpa_file output_path [btree_node_size CompactFormat]." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    unsigned short btree_node_size = ENTRIES_PER_NODE;
    bool CompactFormat = false;

    if (argc == 5) {
        btree_node_size = atoi(argv[3]);
        CompactFormat = atoi(argv[3]);
    } else {
        
    }
    //Create the LM
    LM lm;
    createTrieArray(argv[1], btree_node_size, lm);
    lm.writeBinary(argv[2], CompactFormat);
    return 0;
}
