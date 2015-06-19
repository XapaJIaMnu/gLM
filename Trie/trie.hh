#include "btree.hh"
#include "tokenizer.hh"
#include <limits>
#include <exception>

class offsetTooBig: public std::exception
{
  virtual const char* what() const throw()
  {
    return "The offset is too big to fit in unsigned int. The trie needs to be sharded.";
  }
} offsetEx;

void throwIfOverfow(unsigned int offset, unsigned int next_level_size) noexcept(false) {
    if ((std::numeric_limits<unsigned int>::max() - offset) < next_level_size) {
        throw offsetEx;
    }
}

void addToTrie(B_tree * root_trie, processed_line ngram, unsigned int max_order, unsigned int max_node_size) {
    B_tree * next_level = root_trie;

    //Find the appropriate Btree for insertion. We only insert the last element from ngrams
    //As the previous ones would have been inserted according to the definition of the ngram language model
    for (unsigned int i = 0; i < ngram.ngrams.size() - 1; i++) {
        std::pair<B_tree_node *, int> result = next_level->find_element(ngram.ngrams[i]);
        next_level = result.first->words[result.second].next_level; //Get the next level btree node.
    }

    Entry to_insert;
    if (max_order != ngram.ngram_size) {
        //Create a btree that is going to host the future level.
        B_tree * future_level = new B_tree(max_node_size);

        //Now populate the entry
        to_insert.value = ngram.ngrams.back();
        to_insert.next_level = future_level;
        to_insert.prob = ngram.score;
        to_insert.backoff = ngram.backoff;  //Careful about the <unk> case
    } else {
        //We are at highest order ngram of the model. No next level and no backoff
        to_insert.value = ngram.ngrams.back();
        to_insert.next_level = nullptr;
        to_insert.prob = ngram.score;
        to_insert.backoff = 0.0;
    }

    //Now insert the entry
    next_level->insert_entry(to_insert);
}

size_t calculateTrieSize(B_tree * root_trie) {
    //Returns the total trie size in bytes.
    size_t ret = 0;
    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while(!btrees_to_explore.empty()) {
        B_tree * current = btrees_to_explore.front();
        ret += current->getTotalTreeSize(true /*calculate indexes instead of pointers*/);
        btrees_to_explore.pop(); //We have processed the element, pop it.

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current->root_node);
        do {
            Entry * entry = iter->get_entry();
            if (entry->next_level) {
                btrees_to_explore.push(entry->next_level);
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;
    }
    return ret;
}

//Compresses all of the btree os the tries
void compressTrie(B_tree * root_trie) {
    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while(!btrees_to_explore.empty()) {
        B_tree * current = btrees_to_explore.front();
        btrees_to_explore.pop(); //We have processed the element, pop it.

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current->root_node);
        do {
            Entry * entry = iter->get_entry();
            if (entry->next_level) { //The second check shouldn't be necessary, except for unk! Investigate!
                if (entry->next_level->size == 0) {
                    //Purge empty btrees. Not sure how we get them that's happening though... Maybe because of UNK?
                    delete entry->next_level;
                    entry->next_level = nullptr;
                } else {
                    btrees_to_explore.push(entry->next_level);
                }
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;
        current->compress();
    }
}

//Clears the Trie from memory.
void deleteTrie(B_tree * root_trie) {
    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while(!btrees_to_explore.empty()) {
        B_tree * current = btrees_to_explore.front();
        btrees_to_explore.pop(); //We have processed the element, pop it.

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current->root_node);
        do {
            if (current->size == 0) {
                break; //Btrees with size of 0 don't span other btrees
            }
            Entry * entry = iter->get_entry();
            if (entry->next_level) { //The second check shouldn't be necessary, except for unk! Investigate!
                btrees_to_explore.push(entry->next_level);
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;
        delete current;
    }
}

std::pair<Entry, unsigned short> findNgram(B_tree * root_trie, std::vector<unsigned int> ngrams) {
    //Returns the Entry and the order of the model found
    //We expect given an ngram a b c d to search for P(d|a b c) and the input vector should be [d, c, b, a]
    //For testing correct behaviour on the CPU
    B_tree * next_level = root_trie;
    Entry ret;
    unsigned short level = 0; //We couldn't even find the  first ngram

    for (auto vocabID : ngrams) {
        std::pair<B_tree_node *, int> result = next_level->find_element(vocabID);
        //Check if we have indeed found that element)
        if (result.first->words[result.second] == vocabID) {
            //We have indeed found it
            next_level = result.first->words[result.second].next_level;
            level++;
            ret = result.first->words[result.second];
        } else {
            break; //We didn't find anything, return the last found backoff
        }
    }

    return std::pair<Entry, unsigned short>(ret, level);
}

//Convert the whole Trie to a byte array. Throws if offset becomes too big.
void trieToByteArray(std::vector<unsigned char>& byte_arr, B_tree * root_trie) noexcept(false) {
    bool pointer2Index = true; //We are converting the B_tree * to offset index.
    unsigned int offset = root_trie->getTotalTreeSize(pointer2Index); //Offset from the start of the the array to the desired element

    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while (!btrees_to_explore.empty()) {
        B_tree * current_level = btrees_to_explore.front();

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current_level->root_node);
        do {
            Entry * entry = iter->get_entry();
            if (entry->next_level) {
                entry->offset = offset;  //Set the offset here

                //Throw if overflow
                unsigned int next_level_size = entry->next_level->getTotalTreeSize(pointer2Index);
                throwIfOverfow(offset, next_level_size);

                //We didn't throw, proceed as usual.
                offset+= entry->next_level->getTotalTreeSize(pointer2Index);
                btrees_to_explore.push(entry->next_level);
            } else {
                entry->next_level = 0; //When we don't have a child it's offset is 0
                entry->offset = 0;
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;

        //Convert the trie level to byte array
        current_level->toByteArray(byte_arr, pointer2Index);

        btrees_to_explore.pop();
    }
}

std::pair<Entry, unsigned short> search_byte_array_trie(std::vector<unsigned char>& byte_arr, std::vector<unsigned int> ngrams) {
    unsigned short level = 0; //We couldn't even find the  first ngram
    unsigned int next_btree_start_idx = 0;
    Entry ret;

    for (auto vocabID : ngrams) {
        std::pair<bool, Entry> result = search_byte_arr(byte_arr, vocabID, true /*pointer2Index*/, next_btree_start_idx);
        if (result.first) {
            ret = result.second;
            next_btree_start_idx = result.second.offset;
            level++;
        } else {
            break; //We didn't find the vocabIDs that we were looking, return the longest found previous match
        }
    }

    std::pair<Entry, unsigned short> res(ret, level);
    return res;
}

template<class StringType>
std::pair<B_tree *, unsigned short> createTrie(const StringType infile, unsigned short btree_node_size) {
    //Constructs a trie from an infile and then checks if it can find every token.
    ArpaReader pesho(infile);
    processed_line text = pesho.readline();
    B_tree * root_btree = new B_tree(btree_node_size);

    while (!text.filefinished){
        addToTrie(root_btree, text, pesho.max_ngrams, btree_node_size);
        text = pesho.readline();
    }

    //Btree constructed. Compress it. This is necessary to get rid of empty btrees.
    compressTrie(root_btree);

    //Burden of free is on the calling function. Return the btree and the ngram size
    return std::pair<B_tree *, unsigned short>(root_btree, pesho.max_ngrams);
}

//Create a byte array trie from filename. We are given the byte_arr and we populate it. We also get a metadata file.
template<class StringType>
LM_metadata createTrieArray(const StringType infile, unsigned short btree_node_size, std::vector<unsigned char>& byte_arr){
    std::pair<B_tree *, unsigned short> Trie_and_max_ngram_size = createTrie(infile, btree_node_size);
    B_tree * root_btree = Trie_and_max_ngram_size.first;
    LM_metadata metadata;
    metadata.api_version = 1.0;
    metadata.max_ngram_order = Trie_and_max_ngram_size.second;
    metadata.btree_node_size = btree_node_size;

    //Create a byte array from it;
    size_t trie_size = calculateTrieSize(root_btree);
    byte_arr.reserve(trie_size);
    trieToByteArray(byte_arr, root_btree);
    metadata.byteArraySize = trie_size;

    //Free the B tree
    deleteTrie(root_btree);

    return metadata;
}

template<class StringType>
std::pair<bool, std::string> test_byte_array_trie(const StringType infile, unsigned short btree_node_size) {
    B_tree * root_btree = createTrie(infile, btree_node_size).first;

    //Create a byte array from it;
    size_t trie_size = calculateTrieSize(root_btree);
    std::vector<unsigned char>byte_arr;
    byte_arr.reserve(trie_size);
    trieToByteArray(byte_arr, root_btree);

    //Delete the initial trie;
    deleteTrie(root_btree);

    //Test if everything is there and can be found.
    ArpaReader pesho2(infile);
    processed_line text2 = pesho2.readline();
    bool correct = true;
    std::stringstream error;

    //Check if byte array size is according to the predicted size:
    if (byte_arr.size() != trie_size) {
        error << "Wrong predicted size! Expected " << trie_size << " Actual: " << byte_arr.size() << std::endl;
        correct = false;
    }

    //Now search for every single entry.
    while (!text2.filefinished && correct) {
        std::pair<Entry, unsigned short> found = search_byte_array_trie(byte_arr, text2.ngrams);

        if (found.second) {
            correct = found.first.prob == text2.score;
            correct = correct && (found.first.backoff == text2.backoff);
        } else {
            error << "Ngram not found! " << text2 << std::endl;
            correct = false;
            break;
        }
        if (!correct) {
            error << text2 << std::endl;
            error << "Ngram size is: " << text2.ngram_size << std::endl;
            error << "There has been an error! Score: Expected " << text2.score
            << " Got: " << found.first.prob << " Backoff expected: " << text2.backoff << " Got: " << found.first.backoff << std::endl;
            break;
        }
        text2 = pesho2.readline();
    }

    return std::pair<bool, std::string>(correct, error.str());
}

template<class StringType>
std::pair<bool, std::string> test_trie(const StringType infile, unsigned short btree_node_size) {
    //Constructs a trie from an infile and then checks if it can find every token.
    B_tree * root_btree = createTrie(infile, btree_node_size).first;

    ArpaReader pesho2(infile);
    processed_line text2 = pesho2.readline();
    bool correct = true;
    std::stringstream error;

    //Now search for every single entry.
    while (!text2.filefinished && correct) {
        std::pair<Entry, unsigned short> found = findNgram(root_btree, text2.ngrams);

        if (found.second) {
            correct = found.first.prob == text2.score;
            correct = correct && (found.first.backoff == text2.backoff);
        } else {
            error << "Ngram not found! " << text2 << std::endl;
            correct = false;
            break;
        }
        if (!correct) {
            error << text2 << std::endl;
            error << "Ngram size is: " << text2.ngram_size << std::endl;
            error << "There has been an error! Score: Expected " << text2.score
            << " Got: " << found.first.prob << " Backoff expected: " << text2.backoff << " Got: " << found.first.backoff << std::endl;
            break;
        }
        text2 = pesho2.readline();
    }

    //Free all the used memory
    deleteTrie(root_btree);

    return std::pair<bool, std::string>(correct, error.str());
}
