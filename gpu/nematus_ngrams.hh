#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

//PythonNDarray bullshite
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
using namespace boost::python;

class NematusLM {
    private:
        LM lm;

        //GPU pointers
        unsigned char * btree_trie_gpu;
        unsigned int * first_lvl_gpu;

        std::vector<float *> memory_tracker;

        void doQueries(std::vector<unsigned int>& queries, float * result_storage, size_t results_start_idx);

    public:
        unsigned int gpuMemLimit;
        unsigned int modelMemoryUsage;
        unsigned int queryMemory;
        unsigned int unktoken;
        size_t lastTotalNumQueries = 0; //Keep track of the length of the results array in the last batch

        //This vector contains the softmax vocabulary in order in gLM vocab format.
        std::vector<unsigned int> softmax_vocab_vec;

        NematusLM(char *, char *, unsigned int, int);

        unsigned short getMaxNumNgrams() {
            return lm.metadata.max_ngram_order;
        }

        std::unordered_map<std::string, unsigned int>& getEncodeMap() {
            return lm.encode_map;
        }
        std::unordered_map<unsigned int, std::string>& getDecodeMap() {
            return lm.decode_map;
        }

        float * processBatch(char * path_to_ngrams_file);

        void freeResultsMemory();

        size_t getLastNumQueries();

        boost::python::object processBatchNDARRAY(char * path_to_ngrams_file, long softmax_size, long sentence_length, long batch_size);
        
        ~NematusLM() {
            freeGPUMemory(btree_trie_gpu);
            freeGPUMemory(first_lvl_gpu);
            freeResultsMemory();
        }
};

NematusLM::NematusLM(char * path, char * vocabFilePath, unsigned int maxGPUMemoryMB, int gpuDeviceID = 0) : lm(path) {
    setGPUDevice(gpuDeviceID);

    //Total GPU memory allowed to use (in MB):
    gpuMemLimit = maxGPUMemoryMB;

    //Create the models
    btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());
    modelMemoryUsage = lm.metadata.byteArraySize/(1024*1024) +  (lm.metadata.intArraySize*4/(1024*1024)); //GPU memory used by the model in MB
    queryMemory = gpuMemLimit - modelMemoryUsage; //How much memory do we have for the queries

    unktoken = lm.encode_map.find(std::string("<unk>"))->second; //find the unk token for easy reuse

    //Read the vocab file
    std::unordered_map<unsigned int, std::string> softmax_vocab;
    readDatastructure(softmax_vocab, vocabFilePath);

    softmax_vocab_vec.reserve(softmax_vocab.size());
    for (unsigned int i = 0; i < softmax_vocab.size(); i++) {
        //We should be using an ordered map but I don't have a template for it. Sue me.
        std::string softmax_order_string = softmax_vocab.find(i)->second;
        if (softmax_order_string == "<s>") {
            continue; //Nematus doesn't predict begin of sentence, so we remove it from vocab.
        }
        auto mapiter = lm.encode_map.find(softmax_order_string);
        if (mapiter != lm.encode_map.end()) {
            softmax_vocab_vec.push_back(mapiter->second);
        } else {
            softmax_vocab_vec.push_back(unktoken);
        }
    }
}

void NematusLM::doQueries(std::vector<unsigned int>& queries, float * result_storage, size_t results_start_idx) {
    unsigned int num_keys = queries.size()/lm.metadata.max_ngram_order; //Get how many ngram queries we have to do
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_keys, &results);

    //Search GPU
    searchWrapper(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy results to host and store them:
    copyToHostMemory(results, &result_storage[results_start_idx], num_keys);

    //Free memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);
}

float * NematusLM::processBatch(char * path_to_ngrams_file) {
    //Read in the ngrams file and convert to gLM vocabIDs
    //We need to replace their UNK with ours, replace BoS with 0s
    std::vector<std::vector<unsigned int> > orig_queries;
    std::ifstream is(path_to_ngrams_file);

    if (is.fail() ){
        std::cerr << "Failed to open file " << path_to_ngrams_file << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    while (getline(is, line)) {
        boost::char_separator<char> sep(" ");
        boost::tokenizer<boost::char_separator<char> > tokens(line, sep);
        boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();

        std::vector<unsigned int> orig_query;
        orig_query.reserve(lm.metadata.max_ngram_order);
        bool bogusNgram = false; //Flag about how many </s> we have seen. We only care
        bool seenEoS = false;    //About the first. if the query has 2 EoSs in a row it is
                              //bogus and we shouldn't score it
        bool seenBoS = false;  //We only want one BoS symbol in our queries.
        while(it != tokens.end() ){
            std::string vocabItem = *it;
            if (vocabItem == "</s>") {
                if (seenEoS) {
                    bogusNgram = true;
                }
                seenEoS = true;
            }

            if (bogusNgram) {
                orig_query.push_back(0);
                orig_query[0] = 0; // A bogus ngram is defined by leading 0 so we need to set it.
            } else if (seenBoS) {
                orig_query.push_back(0); //Only one BoS allowed.
            } else {

                if (vocabItem == "<s>") {
                    seenBoS = true;
                }

                auto mapiter = lm.encode_map.find(vocabItem);
                if (mapiter != lm.encode_map.end()) {
                    orig_query.push_back(mapiter->second);
                } else {
                    orig_query.push_back(unktoken);
                }
            }
            it++;
        }
        assert(orig_query.size() == lm.metadata.max_ngram_order); //Sanity check
        orig_queries.push_back(orig_query);
    }

    //Close the stream after we are done.
    is.close();

    //Now we need to expand the queries. Basically every lm.metadata.max_ngram_order word (starting from the first)
    //needs to be replaced byt the full softmax layer
    std::cout << "Total memory required: " << orig_queries.size()*softmax_vocab_vec.size()*lm.metadata.max_ngram_order*4/(1024*1024) << " MB." << std::endl;
    std::cout << "We are allowed to use " << gpuMemLimit << "MB out of which the model takes: " << modelMemoryUsage << "MB leaving us with: "
    << queryMemory << "MB to use for queries." << std::endl;
    //Actually we can use a bit less than queryMemory for our queries, because we need to allocate an array on the GPU that will hold the results, so 
    //We need to calculate that now. Results memory is 1/max_ngram_order of the queryMemory (one float for every max_ngram_order vocab IDs)
    unsigned int queries_memory = (queryMemory*lm.metadata.max_ngram_order)/(lm.metadata.max_ngram_order + 1);
    unsigned int results_memory = queryMemory - queries_memory;
    std::cout << "Query memory: " << queries_memory << "MB. Results memory: " << results_memory << "MB." << std::endl;

    std::vector<unsigned int> all_queries;
    all_queries.reserve((queries_memory*1024*1024 +4)/4);
    size_t total_num_queries = orig_queries.size()*softmax_vocab_vec.size();
    float * all_results = new float[total_num_queries]; //Allocate all results vector
    size_t results_start_idx = 0; //The current index of the latest results.

    //bool first = true; For debugging

    for(auto orig_query : orig_queries) {
        for (unsigned int softmax_vocab_id : softmax_vocab_vec) {
            assert(softmax_vocab_id != 0); //Sanity check
            if (orig_query[0] == 0) { //If the 0th ngram is 0, then this is a bogus query, so keep it that way.
                all_queries.push_back(0);
            } else {
                all_queries.push_back(softmax_vocab_id);
            }
            for (unsigned int i = 1; i < orig_query.size(); i++) { //The first is updated, the rest are the same
                all_queries.push_back(orig_query[i]);
            }
            assert(all_queries.size() % lm.metadata.max_ngram_order == 0); //Sanity check
        }
        if (all_queries.size()*4/(1024*1024) >= queries_memory) {
            //Flush the vector, send the queries to the GPU and write them to a file.
            doQueries(all_queries, all_results, results_start_idx);
            results_start_idx += all_queries.size()/lm.metadata.max_ngram_order; //Update the start index for the next set of results.

            all_queries.clear(); //
            all_queries.reserve((queries_memory*1024*1024 +4)/4);
        }
/* debugging
        if (first) {
            int counter = 0;
            for (int n = 0; n < softmax_vocab_vec.size(); n++) {
                for (int t = 0; t < 6; t++) {
                    std::cout << all_queries[counter] << " ";
                    counter++;
                }
                std::cout << std::endl;
            }
            first = false;
        }
        */
    }
    //Do the last batch:
    doQueries(all_queries, all_results, results_start_idx);
    lastTotalNumQueries = total_num_queries;

    std::cout << "First ten entries: " << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << all_results[i] << " ";
    }
    std::cout << std::endl;

    memory_tracker.push_back(all_results);

    return all_results;

}

void NematusLM::freeResultsMemory() {
    for (auto item : memory_tracker) {
        delete[] item;
    }
    memory_tracker.clear();
}

size_t NematusLM::getLastNumQueries() {
    return lastTotalNumQueries;
}

// wrap c++ array as numpy array
//From Max http://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython
boost::python::object NematusLM::processBatchNDARRAY(char * path_to_ngrams_file, long softmax_size, long sentence_length, long batch_size) {
    float * result = processBatch(path_to_ngrams_file);

    if ((long)lastTotalNumQueries != softmax_size*sentence_length*batch_size) {
        std::cerr << "ARRAY SIZE MISMATCH! SOMETHING IS VERY WRONG!" << std::endl;
        std::cerr << "Total number of queries: " << lastTotalNumQueries << std::endl;
        std::cerr << "softmax_size: " << lastTotalNumQueries/(float)(sentence_length*batch_size) << std::endl;
    }

    npy_intp shape[2] = {sentence_length*batch_size, softmax_size }; // array size
    PyObject* obj = PyArray_SimpleNewFromData(2, shape, NPY_FLOAT, result);
    /*PyObject* obj = PyArray_New(&PyArray_Type, 1, shape, NPY_FLOAT, // data type
                              NULL, result, // data pointer
                              0, NPY_ARRAY_CARRAY_RO, // NPY_ARRAY_CARRAY_RO for readonly
                              NULL);*/
    handle<> array( obj );
    return object(array);
}
