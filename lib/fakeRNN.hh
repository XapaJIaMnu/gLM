#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

class fakeRNN {
    private:
        LM lm;

        //GPU pointers
        unsigned char * btree_trie_gpu;
        unsigned int * first_lvl_gpu;

        void initGPULM(int gpuDeviceId = 0);
        void deleteGPULM();

        std::unordered_map<unsigned int, unsigned int> marian2glmIDs;
    public:
        int gpuMemLimit; //How much GPU memory in total can we use
        int queryMemory; //How much memory on the GPU can we use for queries

        fakeRNN(std::string, std::string, int, int);
        void batchRNNQuery(std::vector<std::vector<int> >& input, std::vector<float>& output);
        void decodeRNNQuery(std::vector<std::vector<int> >& input, std::vector<float>& output);
        void loadVocab(const std::string& vocabPath);
        ~fakeRNN() {
            deleteGPULM();
        }
};
