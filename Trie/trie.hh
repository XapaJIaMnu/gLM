#include "btree.hh"
#include "tokenizer.hh"

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
        ret += current->getTotalTreeSize();
        btrees_to_explore.pop(); //We have processed the element, pop it.

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current->root_node);
        do {
            Entry * entry = iter->get_entry();
            if (entry->next_level && entry->next_level->size > 0) { //The second check shouldn't be necessary, except for unk! Investigate!
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
                    //Purge empty btrees. Not sure how we get them that's happening though...
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
