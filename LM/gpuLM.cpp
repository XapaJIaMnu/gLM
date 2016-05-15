#include "gpuLM.hh"

void gpuLM::query(float * result, unsigned int * queries, size_t num_queries) {
    size_t num_keys = num_queries*lm.metadata.max_ngram_order;
    copyToGPUMemoryNoAlloc(query_input, queries, num_keys);

    //GPU search now:
    searchWrapper(btree_trie_gpu, first_lvl_gpu, query_input, num_queries, query_output, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    copyToHostMemory(query_output, result, num_queries);
}
