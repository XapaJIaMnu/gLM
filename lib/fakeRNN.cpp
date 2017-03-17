#include "fakeRNN.hh"
#include <yaml-cpp/yaml.h>

fakeRNN::fakeRNN(std::string glmPath, std::string vocabPath, int softmax_size, int gpuDeviceID, int gpuMem)
  : lm(glmPath), engine(1, lm, gpuDeviceID), softmax_layer_size(softmax_size), gpuMemLimit(gpuMem) {

    int modelMemoryUsage = lm.metadata.byteArraySize/(1024*1024) + (lm.metadata.intArraySize*4/(1024*1024)); //GPU memory used by the model in MB

    if (modelMemoryUsage > gpuMemLimit) {
        std::cerr << "Not enough memory to load the language model on the GPU! The model size is " << modelMemoryUsage
        << "MB. but we are only allowed to use " << gpuMemLimit << "MB. Try increasing this limit." << std::endl;
        std::exit(1);
    }

    queryMemory = gpuMemLimit - modelMemoryUsage;
    //Read in vocab from json or yaml and create a map from their IDs to ours, as well as softmax vocab vector
    loadVocab(vocabPath);
}

void fakeRNN::batchRNNQuery(std::vector<size_t>& input, unsigned int batch_size, float * gpuMemoryResults) {
    std::vector<std::vector<unsigned int> > sentences;
    makeSents(input, batch_size, sentences);

    //Make sentences to queries
    std::vector<std::vector<unsigned int> > queries(batch_size); //@TODO direct memory copy here
    for (size_t i = 0; i<batch_size; i++) {
        vocabIDsent2queries(sentences[i], queries[i]);
    }

    //Reshuffle them to the expected order, batch by batch and not sentences by sentences
    std::vector<std::vector<unsigned int> >queries_in_batch;
    queries_in_batch.resize(input.size()*lm.metadata.max_ngram_order);
    for (size_t i = 0; i<input.size()/batch_size; i++) {
        for (size_t j = 0; j<batch_size; j++) {
            queries_in_batch[i+j].reserve(lm.metadata.max_ngram_order);
            for (unsigned int t = 0; t<lm.metadata.max_ngram_order; t++) {
                queries_in_batch[i+j].push_back(queries[j][i+t]);
            }
        }
    }

    //Now expand every single query with the softmax layer size @TODO do that in the previous step
    size_t EoS = lm.encode_map.find("</s>")->second;
    std::vector<unsigned int> all_queries;
    all_queries.reserve(input.size()*lm.metadata.max_ngram_order*softmax_layer.size());
    for (std::vector<unsigned int> query : queries_in_batch) {
        for (size_t i = 0; i < softmax_layer.size(); i++) {
            if (query[0] == EoS && query[1] == EoS) { //if a query contains two consecutive EoS (</s>) symbols it means it should be discarded
                                                      //because that's just the section of padded zeroes at the end to be ignored.
                for (size_t j = 0; j<query.size(); j++) { //@TODO memcpy
                    all_queries.push_back(0); //queries with leading zeroes will be ignored
                }
            } else {
                all_queries.push_back(softmax_layer[i]);
                for (size_t j = 1; j<query.size(); j++) {
                    all_queries.push_back(query[j]);
                }
            }
        }
    }

    //now dispatch those to the GPU
    //@TODO obey memory limits
    unsigned int num_keys = all_queries.size()/lm.metadata.max_ngram_order; //Get how many ngram queries we have to do
    if (queryMemory < (int)(all_queries.size()*4/(1024*1024))) {
        unsigned int * gpuKeys = copyToGPUMemory(all_queries.data(), all_queries.size());

        //Search GPU
        engine.search(gpuKeys, num_keys, gpuMemoryResults, 0);
        freeGPUMemory(gpuKeys);
    } else {
        //Estimate at most how many keys we can fit in one query on the GPU
        unsigned int estimate_keys = (queryMemory*1024*1024)/(lm.metadata.max_ngram_order*4);
        unsigned int queried_so_far = 0;
        unsigned int remaining_queries = num_keys;
        while (queried_so_far != num_keys) {
            //Determine how many we are going to run on this batch
            if (remaining_queries < estimate_keys) {
                estimate_keys = remaining_queries;
            }
            queried_so_far += estimate_keys;
            remaining_queries -= estimate_keys;
            unsigned int * gpuKeys = copyToGPUMemory(all_queries.data()+(queried_so_far*lm.metadata.max_ngram_order*4), estimate_keys*lm.metadata.max_ngram_order);

            //Search GPU
            engine.search(gpuKeys+queried_so_far, estimate_keys, gpuMemoryResults, 0);
            freeGPUMemory(gpuKeys);
        }
    }

    
}

void fakeRNN::decodeRNNQuery(std::vector<std::vector<int> >& input, unsigned int batch_size, float * gpuMemoryResults) {
    std::cerr << "stub! decodenRNNQuery not implemented yet!" << std::endl;
}

void fakeRNN::makeSents(std::vector<size_t>& input, unsigned int batch_size, std::vector<std::vector<unsigned int> >& proper_sents) {
    //Some memory preallocation
    size_t BoS = lm.encode_map.find("<s>")->second;
    proper_sents.reserve(batch_size);
    for (auto vec : proper_sents) {
        vec.resize(input.size()/batch_size + 1);//We need a BoS token
        vec[0] = BoS;
    }

    //Convert the sentences including
    for (size_t i = 0; i < input.size()/batch_size; i++) {
        for (size_t j = 0; j<batch_size; j++) {
            proper_sents[j][i + 1] = marian2glmIDs.find(input[i + batch_size*j])->second; //We always start with BoS although we don't care for it
        }
    }
}

void fakeRNN::vocabIDsent2queries(std::vector<unsigned int>& vocabIDs, std::vector<unsigned int>& ret) {
    int ngram_order = lm.metadata.max_ngram_order;
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

}

void fakeRNN::loadVocab(const std::string& vocabPath) {
    std::ifstream ifs;
    ifs.open (vocabPath, std::ifstream::in);
    if (!ifs)   {
        throw std::runtime_error("Couldn't open stream at " + vocabPath);
    }
    //Define unk
    unsigned int ourUNKid = lm.encode_map.find("<unk>")->second;
    YAML::Node vocab = YAML::Load(ifs);
    for (auto&& pair : vocab) {
        auto str = pair.first.as<std::string>();
        auto theirID = pair.second.as<size_t>();

        //map words to gLM vocab
        if (str == "<s>") {
            //BoS token is not used, skip it
            continue;
        }

        if (str == "UNK") {
            str = "<unk>"; //Thier unk to our unk
        }

        if (theirID < softmax_layer_size) {
            //Find the position in our map
            auto mapiter = lm.encode_map.find(str);
            if (mapiter == lm.encode_map.end()) { //If we can't find the word map their ID to UNK
                marian2glmIDs.insert(std::pair<size_t, unsigned int>(theirID, ourUNKid));
            } else {
                marian2glmIDs.insert(std::pair<size_t, unsigned int>(theirID, mapiter->second));
            }
        } else {
            //We have reached vocabulary cutoff, anything else just gets mapped to unk
            marian2glmIDs.insert(std::pair<size_t, unsigned int>(theirID, ourUNKid));
        }
    }
    ifs.close();

    //Now populate the softmax_layer_vector, which is just our vocabIDs sorted by theirID order
    softmax_layer.reserve(softmax_layer_size);
    for (size_t i = 0; i < softmax_layer_size; i++) {
        softmax_layer.push_back(marian2glmIDs.find(i)->second);
    }
}
