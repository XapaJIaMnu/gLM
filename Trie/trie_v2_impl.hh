#include "trie_v2.hh"

template<class StringType>
void createTrie(const StringType filename, LM& lm, unsigned short BtreeNodeSize) {
    //Initialize the LM datastructure
    lm.metadata.api_version = API_VERSION;
    lm.metadata.btree_node_size = BtreeNodeSize;
    //The first 4 bytes of the Btree byte array should be empty in order to use "0" as invalid start value for any
    //next_level field.
    lm.trieByteArray.resize(sizeof(unsigned int), 0);

    //Open the arpa file
    ArpaReader arpain(filename);
    processed_line text;

    //Some info about BTree stumps
    std::vector<size_t> stumps(arpain.max_ngrams - 1, 0); //Unigrams don't have stumps
    std::vector<size_t> total_btrees(arpain.max_ngrams - 1, 0);

    //First level is just an array. Lay the elements as: next_level, prob, backoff. VocabID is a function of the index of the array
    do {
        text = arpain.readline();
        //Store in the following manner: next_level, prob, backoff, next_level...
        size_t current_size = lm.first_lvl.size();
        lm.first_lvl.resize(current_size + 3, 0); //Resize to accomodate the four elements necessary for this entry and initialize them to 0
        std::memcpy(&lm.first_lvl[current_size + 1], &text.score, sizeof(text.score)); //prob
        std::memcpy(&lm.first_lvl[current_size + 2], &text.backoff, sizeof(text.backoff)); //VocabID
    } while (text.ngram_size == 1 && !text.filefinished);

    /*Subsecuent levels except the last one are all the same:
     1) Read in all ngrams from the order.
     2) Sort them by prefix (or suffix because they are reversed).
     3) Add them group by group to the BTrees, in a sorted manner
    */
    
    unsigned short current_ngram_size = 2;

    while (!text.filefinished) {
        std::vector<processed_line> ngrams;
        while (text.ngram_size == current_ngram_size && !text.filefinished) {
            ngrams.push_back(text);
            text = arpain.readline();
        }

        bool lastNgram = false;
        if (text.filefinished) {
            lastNgram = true;
        }

        //sort the ngrams
        std::sort(ngrams.begin(), ngrams.end());

        //Create a BTree from each of them
        processed_line prev_line = ngrams[0]; //Fake the previous line to be the first line. 
                                              //This makes the if statement 11 lines later pass on the first iteration.

        std::vector<unsigned int> context(current_ngram_size - 1);
        std::vector<Entry_v2> entries_to_insert;
        for (auto ngram : ngrams) {
            //Create an entry
            unsigned int vocabID = ngram.ngrams[current_ngram_size - 1];
            float prob = ngram.score;
            float backoff = ngram.backoff;
            Entry_v2 entry = {vocabID, prob, backoff};

            //Insert entries in vector only if the current prefix is equal to the previous.
            if (std::equal(ngram.ngrams.begin(), ngram.ngrams.begin() + current_ngram_size - 1, prev_line.ngrams.begin())) {
                entries_to_insert.push_back(entry);
                //Create a context. The context is everything minus the last word of the ngram vector
                std::copy(ngram.ngrams.begin(), ngram.ngrams.begin() + current_ngram_size - 1, context.begin());
            } else {
                //Add to the BtreeTrie
                total_btrees[current_ngram_size - 2]++;
                if (entries_to_insert.size() <= BtreeNodeSize) {
                    stumps[current_ngram_size - 2]++;
                }
                addBtreeToTrie(entries_to_insert, lm.trieByteArray, lm.first_lvl, context, BtreeNodeSize, lastNgram);
                entries_to_insert.clear();
                entries_to_insert.push_back(entry);
                //Create a context in case of only a single ngram with this context
                std::copy(ngram.ngrams.begin(), ngram.ngrams.begin() + current_ngram_size - 1, context.begin());
            }

            prev_line = ngram; //Keep track of the previous line
        }
        //Handle the last case. Take the last entry from the ngrams vector
        std::copy(ngrams[ngrams.size() - 1].ngrams.begin(), ngrams[ngrams.size() - 1].ngrams.begin() + current_ngram_size - 1, context.begin());
        total_btrees[current_ngram_size - 2]++;
        if (entries_to_insert.size() <= BtreeNodeSize) {
            stumps[current_ngram_size - 2]++;
        }
        addBtreeToTrie(entries_to_insert, lm.trieByteArray, lm.first_lvl, context, BtreeNodeSize, lastNgram);
        entries_to_insert.clear();

        current_ngram_size++;
    }

    //Add some data to the lm datastructure:
    lm.metadata.max_ngram_order = arpain.max_ngrams;
    lm.metadata.byteArraySize = lm.trieByteArray.size();
    lm.metadata.intArraySize = lm.first_lvl.size();
    lm.encode_map = arpain.encode_map;
    lm.decode_map = arpain.decode_map;

    //Print stumps statistics:
    for (unsigned int i = 0; i < stumps.size(); i++) {
        std::cout << "There are: " << stumps[i] << " stumps among " << i + 2 << "grams out of " <<
        total_btrees[i] << " BTrees in total, " << ((double)stumps[i]/(double)total_btrees[i])*100 << " % of all."<< std::endl;
    }

}

void addBtreeToTrie(std::vector<Entry_v2> &entries_to_insert, std::vector<unsigned char> &byte_arr, std::vector<unsigned int> &first_lvl,
 std::vector<unsigned int> context, unsigned short BtreeNodeSize, bool lastNgram) {
    /*
    1) Find the current context.
    2) Set the next_level to the current byte_arr end.
    3) create a Btree at the end of the byte_arr
    */
    
    //Since we are looking for the context in the trie, the lastNgram variable in this call is always false
    Entry_with_offset cur_context = searchTrie(byte_arr, first_lvl, context, BtreeNodeSize, false);

    //Check for buggy arpa files
    if (!cur_context.found) {
        std::cerr << "Could not find a lower order ngram even though a higher order one exists!" << std::endl;
        std::cerr << "Context that wasn't found: ";
        for (auto word : context) {
            std::cerr << word << ' ';
        }
        std::cerr << std::endl << "Please rebuild the ARPA file using lmplz from KenLM." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    //Assign the next_level for this context.
    assert((byte_arr.size() - cur_context.currentBtreeStart) % 4 == 0); //Sanity check.
    *cur_context.next_level = (byte_arr.size() - cur_context.currentBtreeStart)/4;

    //create a Btree at the next level
    array2balancedBtree(byte_arr, entries_to_insert, BtreeNodeSize, lastNgram);

}

Entry_with_offset searchTrie(std::vector<unsigned char> &btree_trie_byte_arr, std::vector<unsigned int> &first_lvl,
    std::vector<unsigned int> ngrams, unsigned short BtreeNodeSize, bool lastNgram) {

    //sanity check
    assert(ngrams[0] <= first_lvl.size());

    //First level search is easy -> the next_level, prob and backoff for vocabID n are located at (n-1)*3, (n-1)*3+1 and (n-1)*3+2 of the
    //byte_arr 
    unsigned int * next_level = &first_lvl[(ngrams[0]-1)*3];
    float * prob = reinterpret_cast<float *>(&first_lvl[(ngrams[0]-1)*3 + 1]);
    float * backoff = reinterpret_cast<float *>(&first_lvl[(ngrams[0]-1)*3 + 2]);
    unsigned int vocabID = ngrams[0];

    struct Entry_with_offset entry_traverse = {
        vocabID, //VocabID
        next_level, //unsigned int * next_level
        *prob,
        *backoff,
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

template<class StringType>
std::pair<bool, std::string> test_trie(const StringType filename, unsigned short BtreeNodeSize) {
    LM lm;
    createTrie(filename, lm, BtreeNodeSize);

    //Try to find every ngram:
    ArpaReader infile(filename);
    processed_line text = infile.readline();
    bool correct = true;
    std::stringstream error;

    while (!text.filefinished) {
        bool lastNgram = false;
        if (text.ngram_size == infile.max_ngrams) {
            lastNgram = true;
        }
        Entry_with_offset res = searchTrie(lm.trieByteArray, lm.first_lvl, text.ngrams, BtreeNodeSize, lastNgram);

        if (!res.found) {
            error << "Couldn't find entry " << text << std::endl;
            correct = false;
            break;
        } else if (res.prob != text.score) {
            error << "Expected probability: " << text.score << ", got: " << res.prob << std::endl << text;
            correct = false;
            break;
        } else if (!lastNgram && res.backoff != text.backoff) {
            error << "Expected backoff: " << text.backoff << ", got: " << res.backoff << std::endl << text;
            correct = false;
            break;
        }
        text = infile.readline();
    }

    return std::pair<bool, std::string>(correct, error.str());
}

template<class StringType>
std::pair<bool, std::string> test_trie(LM &lm, const StringType filename, unsigned short BtreeNodeSize) {

    //Try to find every ngram:
    ArpaReader infile(filename);
    processed_line text = infile.readline();
    bool correct = true;
    std::stringstream error;

    while (!text.filefinished) {
        bool lastNgram = false;
        if (text.ngram_size == infile.max_ngrams) {
            lastNgram = true;
        }
        Entry_with_offset res = searchTrie(lm.trieByteArray, lm.first_lvl, text.ngrams, BtreeNodeSize, lastNgram);

        if (!res.found) {
            error << "Couldn't find entry " << text << std::endl;
            correct = false;
            break;
        } else if (res.prob != text.score) {
            error << "Expected probability: " << text.score << ", got: " << res.prob << std::endl << text;
            correct = false;
            break;
        } else if (!lastNgram && res.backoff != text.backoff) {
            error << "Expected backoff: " << text.backoff << ", got: " << res.backoff << std::endl << text;
            correct = false;
            break;
        }
        text = infile.readline();
    }

    return std::pair<bool, std::string>(correct, error.str());
}
