#include "trie_v2_impl.hh"
#include "lm_impl.hh"
#include "gpu_search.hh"

int main(int argc, char* argv[]){
    if (argc != 5 && argc != 3 && argc != 4) {
        std::cerr << "Usage:" << std::endl << argv[0] << " path_to_arpa_file output_path [btree_node_size]." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    unsigned short btree_node_size = 33;

    if (argc == 4) {
        btree_node_size = atoi(argv[3]);
    }
    //Create the LM
    LM lm;
    createTrie(argv[1], lm, btree_node_size);
    lm.writeBinary(argv[2]);
    return 0;
}
