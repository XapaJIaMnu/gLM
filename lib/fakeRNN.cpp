#include "fakeRNN.hh"
#include <yaml-cpp/yaml.h>

fakeRNN::fakeRNN(std::string glmPath, std::string vocabPath, int gpuDeviceID, int gpuMem) : lm(glmPath), gpuMemLimit(gpuMem) {
    initGPULM(gpuDeviceID);
    //Read in vocab from json or yaml and create a map from their IDs to ours
    loadVocab(vocabPath);
}

void fakeRNN::initGPULM(int gpuDeviceID) {
    setGPUDevice(gpuDeviceID);

    //Create the models
    btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());
    int modelMemoryUsage = lm.metadata.byteArraySize/(1024*1024) +  (lm.metadata.intArraySize*4/(1024*1024)); //GPU memory used by the model in MB
    queryMemory = gpuMemLimit - modelMemoryUsage;

}

void fakeRNN::batchRNNQuery(std::vector<std::vector<int> >& input, std::vector<float>& output) {

}

void fakeRNN::decodeRNNQuery(std::vector<std::vector<int> >& input, std::vector<float>& output) {

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
        auto theirID = pair.second.as<unsigned int>();

        //map words to gLM vocab
        if (str == "<s>") {
            //BoS token is not used, skip it
            continue;
        }

        if (str == "UNK") {
            str = "<unk>"; //Thier unk to our unk
        }

        //Find the position in our map
        auto mapiter = lm.encode_map.find(str);
        if (mapiter == lm.encode_map.end()) { //If we can't find the word map their ID to UNK
            marian2glmIDs.insert(std::pair<unsigned int, unsigned int>(theirID, ourUNKid));
        } else {
            marian2glmIDs.insert(std::pair<unsigned int, unsigned int>(theirID, mapiter->second));
        }
    }
    ifs.close();
}
