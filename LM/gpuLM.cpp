#include "gpuLM.hh"

void gpuLM::query(float * result, unsigned int * queries, size_t queries_length) {
    unsigned int num_keys = queries_length/lm.metadata.max_ngram_order;
    copyToGPUMemoryNoAlloc(query_input, queries, queries_length);

    //GPU search now:
    searchWrapper(btree_trie_gpu, first_lvl_gpu, query_input, num_keys, query_output, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    copyToHostMemory(query_output, result, num_keys);
}
