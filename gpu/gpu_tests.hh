#include "memory_management.hh"
#include "gpu_search.hh"
#include "trie.hh"
#include "structs.hh"
#include <sstream>

template<class StringType>
std::pair<bool, std::string> testQueryNgrams(StringType arpafile) {
    //Create the models
    LM lm;
    createTrieArray(arpafile, 127, lm); //@Todo make this somewhat not so hardcoded
    std::cout << "Finished create Btree trie." << std::endl;
    //Create check against things:
    ArpaReader pesho2(arpafile);
    processed_line text2 = pesho2.readline();

    unsigned int num_keys = 0; //How many ngrams are we going to query
    std::vector<unsigned int> keys;
    std::vector<float> check_against;
    unsigned short max_ngram_order = lm.metadata.max_ngram_order;

    while (!text2.filefinished) {
        //Inefficient reallocation of keys_to_query. Should be done better
        unsigned int num_padded =  max_ngram_order - text2.ngrams.size();
        for (unsigned int i = 0; i < num_padded; i++) {
            text2.ngrams.push_back(0); //Extend ngrams to max num ngrams if they are of lower order
        }
        
        for (unsigned int i = 0; i < max_ngram_order; i++) {
            keys.push_back(text2.ngrams[i]); //Extend ngrams to max num ngrams if they are of lower order
        }

        check_against.push_back(text2.score);

        num_keys++;
        text2 = pesho2.readline();
    }

    unsigned char * gpuByteArray = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());

    unsigned int * gpuKeys = copyToGPUMemory(keys.data(), keys.size());
    float * results;
    allocateGPUMem(num_keys, &results);

    searchWrapper(gpuByteArray, gpuKeys, num_keys, results);

    //Copy back to host
    float * results_cpu = new float[num_keys];
    copyToHostMemory(results, results_cpu, num_keys);

    //Clear gpu memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(gpuByteArray);
    freeGPUMemory(results);

    bool allcorrect = true;
    std::stringstream error;
    for (unsigned int i = 0; i < num_keys; i++) {
        float res_prob = results_cpu[i];

        float exp_prob = check_against[i];

        if (!(exp_prob == res_prob)) {
            error << "Error expected prob: " << exp_prob << " got: " << res_prob << "."<< std::endl;
            allcorrect = false;
            break;
        }
    }
    delete[] results_cpu;

    return std::pair<bool, std::string>(allcorrect, error.str());

}

//Converts a raw sentence into one suitable for generating ngrams from, with vocabIDs
std::vector<unsigned int> sent2vocabIDs(LM &lm, std::vector<std::string> input) {
    std::vector<unsigned int> ret;
    ret.reserve(input.size() + 2);
    unsigned int unktoken = lm.encode_map.find(std::string("<unk>"))->second; //@TODO don't look up UNKTOKEN every time, get it from somewhere
    unsigned int beginsent = lm.encode_map.find(std::string("<s>"))->second;
    unsigned int endsent = lm.encode_map.find(std::string("</s>"))->second;

    ret.push_back(beginsent);
    for (auto item : input) {
        std::map<std::string, unsigned int>::iterator it = lm.encode_map.find(item);
        if (it != lm.encode_map.end()) {
            ret.push_back(it->second);
        } else {
            ret.push_back(unktoken);
        }
    }
    ret.push_back(endsent);

    return ret;
}

std::vector<unsigned int> vocabIDsent2queries(std::vector<unsigned int> vocabIDs, unsigned short ngram_order) {
    std::vector<unsigned int> ret;
    ret.reserve((vocabIDs.size() - 1)*ngram_order);
    int front_idx = 0;
    int back_idx = 1;

    //In the ret vector put an ngram for every single entry
    while (back_idx < (int)vocabIDs.size()) {
        for (int i = back_idx; i >= front_idx; i--) {
            ret.push_back(vocabIDs[i]);
        }
        //Pad with zeroes if we don't have enough
        int zeroes_to_pad = ngram_order - (back_idx - front_idx) - 1;
        for (int i = 0; i < zeroes_to_pad; i++) {
            ret.push_back(0);
        }

        //determine which ptr to advance
        if ((back_idx - front_idx) < (ngram_order - 1)) {
            back_idx++;
        } else {
            front_idx++;
            back_idx++;
        }
    }

    return ret;
}

std::vector<std::string> interactiveRead() {
    std::string response;
    boost::char_separator<char> sep(" ");
    while (true) {
        getline(std::cin, response);
        if (response == "/end") {
            break;
        }
        std::vector<std::string> sentence;
        boost::tokenizer<boost::char_separator<char> > tokens(response, sep);
        for (auto word : tokens) {
            sentence.push_back(word);
        }
        //Now process the sentences
    }
    return std::vector<std::string>{std::string("pesho")};
}
