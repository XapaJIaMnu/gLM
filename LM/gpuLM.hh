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
        gpuLM(StringType, size_t);
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
