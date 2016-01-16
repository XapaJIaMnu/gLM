#include "btree_v2_impl.hh"
#include "tokenizer.hh"
#include "lm.hh"

template<class StringType>
void createTrie(const StringType filename, LM& lm, unsigned short BtreeNodeSize);
void addBtreeToTrie();
void addBtreeToTrie(std::vector<Entry_v2> &entries_to_insert, std::vector<unsigned char> &byte_arr,
 std::vector<unsigned int> context, unsigned short BtreeNodeSize, bool lastNgram);
