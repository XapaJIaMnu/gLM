#include "structs.hh" 

//A special struct for searching the BTree
struct Entry_with_offset {
    unsigned int vocabID;
    unsigned int * next_level; //ptr to next trie level so that we can set it later
    float prob;
    float backoff;
    //Elements in case we need to go down in the trie
    size_t next_child_offset;
    unsigned int next_child_size;
    //Those two will make debugging easier during development. They add a miniscule overhead so I will just leave them as they are.
    bool found;
    unsigned int found_idx;
    //This is for CPU search
    unsigned int currentBtreeStart;
};

void array2balancedBtree(std::vector<unsigned char> &byte_arr, std::vector<Entry_v2> &array, unsigned short BtreeNodeSize, bool lastNgram);
unsigned int futureSizeCalculator(unsigned int size, unsigned short BtreeNodeSize, int payload_size);
void entry_v2_to_node(std::vector<unsigned char> &byte_arr, std::vector<Entry_v2> &entries, std::vector<unsigned int> offsets, unsigned int payload_size);
std::vector<unsigned int> createEvenSplits(unsigned int array_size, unsigned short BtreeNodeSize);
Entry_with_offset searchBtree(std::vector<unsigned char> &byte_arr, size_t BtreeStartPosition, unsigned short BtreeNodeSize, unsigned int vocabID, bool lastNgram);
std::pair<unsigned int, bool> linearSearch(unsigned int * arr_to_search, unsigned int size, unsigned int vocabID);
Entry_with_offset searchNode(std::vector<unsigned char> &byte_arr, size_t StartPosition, unsigned int node_size, unsigned int vocabID,
 unsigned short payload_size, unsigned short BtreeNodeSize);
