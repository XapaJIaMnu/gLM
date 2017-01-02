#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"

int main(int argc, char* argv[]) {
    if (!(argc != 5 || argc != 6)) {
        std::cerr << "Usage: " << argv[0] << " path_to_model_dir path_to_ngrams_file path_to_vocab_file maxGPUMemory [gpuDeviceID=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 6) {
        setGPUDevice(atoi(argv[5]));
    }
    //Create the models
    LM lm(argv[1]);
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    unsigned int unktoken = lm.encode_map.find(std::string("<unk>"))->second; //find the unk token for easy reuse

    //Read the vocab file
    std::unordered_map<unsigned int, std::string> softmax_vocab;
    readDatastructure(softmax_vocab, argv[3]);

    //This vector contains the softmax vocabulary in order in gLM vocab format.
    std::vector<unsigned int> softmax_vocab_vec(softmax_vocab.size());
    for (unsigned int i = 0; i < softmax_vocab.size(); i++) {
        //We should be using an ordered map but I don't have a template for it. Sue me.
        std::string softmax_order_string = softmax_vocab.find(i)->second;
        auto mapiter = lm.encode_map.find(softmax_order_string);
        if (mapiter != lm.encode_map.end()) {
            softmax_vocab_vec.push_back(mapiter->second);
        } else {
            softmax_vocab_vec.push_back(unktoken);
        }
    }

    //Read in the ngrams file and convert to gLM vocabIDs
    //We need to replace their UNK with ours, replace BoS with 0s
    std::vector<std::vector<unsigned int> > orig_queries;
    std::ifstream is(argv[2]);

    if (is.fail() ){
        std::cerr << "Failed to open file " << argv[2] << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    getline(is, line);
    while (getline(is, line)) {
        boost::char_separator<char> sep(" ");
        boost::tokenizer<boost::char_separator<char> > tokens(line, sep);
        boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();

        std::vector<unsigned int> orig_query(lm.metadata.max_ngram_order);
        while(it != tokens.end() ){
            std::string vocabItem = *it;
            if (vocabItem == "<s>") {
                orig_query.push_back(0);
            } else {
                auto mapiter = lm.encode_map.find(vocabItem);
                if (mapiter != lm.encode_map.end()) {
                    orig_query.push_back(mapiter->second);
                } else {
                    orig_query.push_back(unktoken);
                }
            }
            it++;
        }
        orig_queries.push_back(orig_query);
    }

    //Close the stream after we are done.
    is.close();

    //Now we need to expand the queries. Basically every lm.metadata.max_ngram_order word (starting from the first)
    //needs to be replaced byt the full softmax layer
    std::cout << "Memory required: " << orig_queries.size()*softmax_vocab_vec.size()*lm.metadata.max_ngram_order*4/(1024*1024) << " MB." << std::endl;
    std::vector<unsigned int> all_queries(orig_queries.size()*softmax_vocab_vec.size()*lm.metadata.max_ngram_order);
    all_queries.reserve(orig_queries.size()*softmax_vocab_vec.size()*lm.metadata.max_ngram_order);

    for(auto orig_query : orig_queries) {
        for (unsigned int softmax_vocab_id : softmax_vocab_vec) {
            all_queries.push_back(softmax_vocab_id);
            for (unsigned int i = 1; i < orig_query.size(); i++) { //The first is updated, the rest are the same
                all_queries.push_back(orig_query[i]);
            }
        }
    }

    //Now split it in sections and query it on the GPU

    //Free GPU memory
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);
    return 0;
}
