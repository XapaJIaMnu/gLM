#include "btree_v2.hh"
#include <stdlib.h>
#include <set>
#include <algorithm>

int main(int argc, char* argv[]) {
    unsigned int BtreeNodeSize = 3;
    bool lastNgram = false;
    unsigned int num_entries = 25;

    if (argc == 4) {
        BtreeNodeSize = atoi(argv[1]);
        num_entries = atoi(argv[2]);
        lastNgram = (bool)atoi(argv[3]);
    }

    std::set<unsigned int> prev_nums; //Used to see if we have duplicating nums
    std::vector<Entry_v2> array;
    while (prev_nums.size() < num_entries) {
        unsigned int new_entry = 1 + (rand() % (num_entries*10));
        if (prev_nums.count(new_entry) == 0){
            Entry_v2 new_entry_actual = {new_entry, prev_nums.size() + 0.0f, prev_nums.size() + 0.5f};
            array.push_back(new_entry_actual);
            prev_nums.insert(new_entry);
        }
    }

    //input needs to be sorted for this BTree
    std::sort(array.begin(), array.end()); 

    std::vector<unsigned char> btree_byte_arr;
    array2balancedBtree(btree_byte_arr, array, BtreeNodeSize, lastNgram);
/*
    for (auto entry : array) {
        Entry_with_offset test = searchBtree(btree_byte_arr, 0, BtreeNodeSize, entry.vocabID, lastNgram);
        std::cout << test.vocabID << std::endl;
    }
*/
    std::pair<bool, std::string> res = test_btree_v2(num_entries, BtreeNodeSize, lastNgram);
    if (!res.first) {
        std::cout << res.second;
    }
    return 0;
}
