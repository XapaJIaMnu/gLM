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
        to_insert.backoff = ngram.backoff;
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
