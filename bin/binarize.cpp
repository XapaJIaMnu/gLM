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
    //Create the btree_array;
    std::vector<unsigned char> byte_arr;
    LM_metadata metadata = createTrieArray(argv[1], btree_node_size, byte_arr);
    writeBinary(argv[2], byte_arr, metadata);
    return 0;
}
