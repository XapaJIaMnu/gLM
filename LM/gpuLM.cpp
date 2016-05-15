#include "gpuLM.hh"

void gpuLM::query(float * result, unsigned int * queries, size_t num_queries) {
    size_t num_keys = num_queries*lm.metadata.max_ngram_order;
    copyToGPUMemoryNoAlloc(query_input, queries, num_keys);

    //GPU search now:
    searchWrapper(btree_trie_gpu, first_lvl_gpu, query_input, num_queries, query_output, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    copyToHostMemory(query_output, result, num_queries);
}


QueryMemory::QueryMemory(std::atomic<unsigned int>& threadID, size_t max_num_queries, unsigned short max_ngram_order) {
    stream = threadID.fetch_add(1, std::memory_order_seq_cst); //Set the stream to be used by this thread
    pinnedMemoryAllocator(&results, max_num_queries);
    pinnedMemoryAllocator(&ngrams_for_query, max_num_queries*max_ngram_order);
}

//Single batch, always use the default stream.
QueryMemory::QueryMemory(size_t max_num_queries, unsigned short max_ngram_order) {
    stream = 0;
    pinnedMemoryAllocator(&results, max_num_queries);
    pinnedMemoryAllocator(&ngrams_for_query, max_num_queries*max_ngram_order);
}

QueryMemory::~QueryMemory() {
    pinnedMemoryDeallocator(results);
    pinnedMemoryDeallocator(ngrams_for_query);
}
