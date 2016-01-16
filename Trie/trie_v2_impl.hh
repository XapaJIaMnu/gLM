#include "trie_v2.hh"

template<class StringType>
void createTrie(const StringType filename, LM& lm, unsigned short BtreeNodeSize) {
    //Open the arpa file
    ArpaReader arpain(filename);
    processed_line text;

    //First level is just an array. Lay the elements as: next_level, prob, backoff. VocabID is a function of the index of the array
    do {
        text = arpain.readline();
        //Store in the following manner: next_level, prob, backoff, next_level...
        size_t current_size = lm.trieByteArray.size();
        lm.trieByteArray.resize(current_size + 16, 0); //Resize to accomodate the four elements necessary for this entry and initialize them to 0
        std::memcpy(&lm.trieByteArray[current_size + 4], &text.score, sizeof(text.score)); //prob
        std::memcpy(&lm.trieByteArray[current_size + 8], &text.backoff, sizeof(text.backoff)); //VocabID
    } while (text.ngram_size == 1 && !text.filefinished);

    /*Subsecuent levels except the last one are all the same:
     1) Read in all ngrams from the order.
     2) Sort them by prefix (or suffix because they are reversed).
     3) Add them group by group to the BTrees, in a sorted manner
    */
    
    unsigned short current_ngram_size = 2;
    std::vector<processed_line> ngrams;
    while (text.ngram_size == current_ngram_size && !text.filefinished) {
        //Reverse the ngrams so that they are forward facing
        ngrams.push_back(text);
        text = arpain.readline();
    }

    bool last_ngram_order = false;
    if (text.filefinished) {
        last_ngram_order = true;
    }

    //sort the ngrams
    std::sort(ngrams.begin(), ngrams.end());

    //Create a BTree from each of them
    processed_line prev_line = ngrams[0]; //Fake the previous line to be the first line. 
                                          //This makes the if statement 10 lines later pass on the first iteration.

    std::vector<Entry_v2> entries_to_insert;
    for (auto ngram : ngrams) {
        //Create an entry
        unsigned int vocabID = ngram.ngrams[current_ngram_size - 1];
        float prob = ngram.score;
        float backoff = ngram.backoff;
        Entry_v2 entry = {vocabID, prob, backoff};

        //Insert entries in vector only if the current prefix is equal to the previous.
        if (std::equal(ngram.ngrams.begin(), ngram.ngrams.begin() + current_ngram_size - 2, prev_line.ngrams.begin())) {
            entries_to_insert.push_back(entry);
        } else {
            addBtreeToTrie();
            entries_to_insert.clear();
            entries_to_insert.push_back(entry);
        }

        prev_line = ngram; //Keep track of the previous line
    }
    //Handle the last case:
    addBtreeToTrie();
    entries_to_insert.clear();

}

void addBtreeToTrie(std::vector<Entry_v2> &entries_to_insert, std::vector<unsigned char> &byte_arr,
 std::vector<unsigned int> context, unsigned short BtreeNodeSize, bool lastNgram) {

}

Entry_with_offset searchTrie(std::vector<unsigned char> &btree_trie_byte_arr, std::vector<unsigned int> first_lvl,
    std::vector<unsigned int> ngrams, unsigned short BtreeNodeSize, bool lastNgram) {

    //sanity check
    assert(ngrams[0] <= first_lvl.size());

    //First level search is easy -> the next_level, prob and backoff for vocabID n are located at (n-1)*3, (n-1)*3+1 and (n-1)*3+2 of the
    //byte_arr 
    unsigned int * next_level = &first_lvl[(ngrams[0]-1)*3];
    float prob = first_lvl[(ngrams[0]-1)*3 + 1];
    float backoff = first_lvl[(ngrams[0]-1)*3 + 2];
    unsigned int vocabID = ngrams[0];

    struct Entry_with_offset entry_traverse = {
        vocabID, //VocabID
        next_level, //unsigned int * next_level
        prob,
        backoff,
        0, //size_t next_child_offset; we don't use it
        0, //unsigned short next_child_size; we don't use it
        true, //bool found;
        0, //unsigned int found_idx we don't use it
    };

    if (ngrams.size() == 1) {
        return entry_traverse;
    } else {
        //Search the btree_trie
        size_t current_btree_start = (*entry_traverse.next_level)*4;

        //Check if the last ngram from ngrams is actually located on a final level of a trie
        unsigned int traverse_limit;
        if (lastNgram) {
            traverse_limit = ngrams.size() - 1;
        } else {
            traverse_limit = ngrams.size();
        }

        //perform search
        Entry_with_offset new_entry_traverse;
        for (unsigned int i = 1; i < traverse_limit; i++) {
            new_entry_traverse = searchBtree(btree_trie_byte_arr, current_btree_start, BtreeNodeSize, ngrams[i], false);
            if (new_entry_traverse.found) {
                current_btree_start += (*new_entry_traverse.next_level)*4;
                entry_traverse = new_entry_traverse;
            } else {
                //We didn't find what we were looking for, return the highest order we found but with false
                entry_traverse.found = false;
                return entry_traverse;
            }
        }

        //Now search the lastNgram one
        if (lastNgram) {
            new_entry_traverse = searchBtree(btree_trie_byte_arr, current_btree_start, BtreeNodeSize, ngrams[ngrams.size() - 1], lastNgram);
            if (new_entry_traverse.found) {
                return new_entry_traverse;
            }
        }

        //In case we didn't find the complete ngram or if we found it in the previous for loop but didn't return it, return the correct result now
        return entry_traverse;
    }
}
