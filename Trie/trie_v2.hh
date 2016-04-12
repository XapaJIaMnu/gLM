#include "btree_v2.hh"
#include "tokenizer.hh"
#include "lm.hh"

template<class StringType>
void createTrie(const StringType filename, LM& lm, unsigned short BtreeNodeSize);
void addBtreeToTrie(std::vector<Entry_v2> &entries_to_insert, std::vector<unsigned char> &byte_arr, std::vector<unsigned int> &first_lvl,
 std::vector<unsigned int> context, unsigned short BtreeNodeSize, bool lastNgram);
Entry_with_offset searchTrie(std::vector<unsigned char> &btree_trie_byte_arr, std::vector<unsigned int> &first_lvl,
    std::vector<unsigned int> ngrams, unsigned short BtreeNodeSize, bool lastNgram);

template<class StringType>
std::pair<bool, std::string> test_trie(LM &lm, const StringType filename, unsigned short BtreeNodeSize);
template<class StringType>
std::pair<bool, std::string> test_trie(const StringType filename, unsigned short BtreeNodeSize);
