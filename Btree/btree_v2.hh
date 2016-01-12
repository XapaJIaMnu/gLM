#include "structs.hh" 

void array2balancedBtree(std::vector<unsigned char> &byte_arr, std::vector<Entry_v2> &array, unsigned short BtreeNodeSize, bool lastNgram);
unsigned int futureSizeCalculator(unsigned int size, unsigned short BtreeNodeSize, int payload_size);
void entry_v2_to_node(std::vector<unsigned char> &byte_arr, std::vector<Entry_v2> &entries, std::vector<unsigned int> offsets, unsigned int payload_size);