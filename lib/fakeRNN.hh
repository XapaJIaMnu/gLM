#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

class fakeRNN {
    private:
        LM lm;
        GPUSearcher engine;
        unsigned int softmax_layer_size;
        std::vector<unsigned int> softmax_layer;

        std::unordered_map<size_t, unsigned int> marian2glmIDs;
        void makeSents(std::vector<size_t>& input, unsigned int batch_size, std::vector<std::vector<unsigned int> >& proper_sents);
        void vocabIDsent2queries(std::vector<unsigned int>& vocabIDs, std::vector<unsigned int>& ret);
    public:
        int gpuMemLimit; //How much GPU memory in total can we use
        int queryMemory; //How much memory on the GPU can we use for queries

        /*glmPath, vocabPath, softmax_size, gpuDeviceID, gpuMem, make_exp*/
        fakeRNN(std::string, std::string, int, int, int, bool = true);
        void batchRNNQuery(std::vector<size_t>& input, unsigned int batch_size, float * gpuMemory);
        void decodeRNNQuery(std::vector<std::vector<int> >& input, unsigned int batch_size, float * gpuMemory);
        void loadVocab(const std::string& vocabPath);
};
