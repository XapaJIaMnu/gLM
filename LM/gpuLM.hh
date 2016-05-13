#pragma once

#include "../gpu/gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

class gpuLM {
    private:
        LM lm;

        //GPU pointers
        unsigned char * btree_trie_gpu;
        unsigned int * first_lvl_gpu;

        unsigned int * query_input;
        float * query_output;

    public:
		template<class StringType>
		gpuLM(StringType path, size_t max_num_queries, int gpu_device_id = 0) : lm(path) {
            //Set GPU device
            setGPUDevice(gpu_device_id);

			//Create GPU objects here
			//@TODO remove CPU LM after copy, we don't care about it anymore.
			//@TODO in fact we need to redo the whole LM so it doesn't hog so much memory.
			btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
			first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

			//Allocate max memory input and output queries
			allocateGPUMem(max_num_queries, &query_output);
			allocateGPUMem(max_num_queries*lm.metadata.max_ngram_order, &query_input);
		}

        unsigned short getMaxNumNgrams() {
            return lm.metadata.max_ngram_order;
        }

        std::unordered_map<std::string, unsigned int>& getEncodeMap() {
            return lm.encode_map;
        }
        std::unordered_map<unsigned int, std::string>& getDecodeMap() {
            return lm.decode_map;
        }
        void query(float * result, unsigned int * queries, size_t queries_length);
        ~gpuLM() {
            freeGPUMemory(btree_trie_gpu);
            freeGPUMemory(first_lvl_gpu);
            freeGPUMemory(query_output);
            freeGPUMemory(query_input);
        }
};
