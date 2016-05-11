#include "gpuLM.hh"

template<class StringType>
gpuLM::gpuLM(StringType path, size_t max_num_queries) : lm(path) {
    //Create GPU objects here
    //@TODO remove CPU LM after copy, we don't care about it anymore.
    //@TODO in fact we need to redo the whole LM so it doesn't hog so much memory.
    btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //Allocate max memory input and output queries
    allocateGPUMem(max_num_queries, &query_output);
    allocateGPUMem(max_num_queries*lm.metadata.max_ngram_order, &query_input);
}

void gpuLM::query(float * result, unsigned int * queries, size_t queries_length) {
    unsigned int num_keys = queries_length/lm.metadata.max_ngram_order;
    copyToGPUMemoryNoAlloc(query_input, queries, queries_length);

    //GPU search now:
    searchWrapper(btree_trie_gpu, first_lvl_gpu, query_input, num_keys, query_output, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    copyToHostMemory(query_output, result, num_keys);
}
