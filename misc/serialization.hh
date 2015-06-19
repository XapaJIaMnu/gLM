#include <iostream>
#include <fstream>
#include <string>
#include "structs.hh"

//Use boost to serialize the vectors
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#define API_VERSION 1.0

template<class StringType>
void SerializeByteArray(std::vector<unsigned char>& byte_arr, const StringType path){
    std::ofstream os (path, std::ios::binary);
    boost::archive::text_oarchive oarch(os);
    oarch << byte_arr;
    os.close();
}

//The byte_arr vector should be reserved to prevent constant realocation.
template<class StringType>
void ReadByteArray(std::vector<unsigned char>& byte_arr, const StringType path){
    std::ifstream is (path, std::ios::binary);

    if (is.fail() ){
        std::cerr << "Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    boost::archive::text_iarchive iarch(is);
    iarch >> byte_arr;
    is.close();
}

template<class StringType>
void storeConfigFile(LM_metadata metadata, const StringType path) {
    std::ofstream configfile(path);

    if (configfile.fail()) {
        std::cerr << "Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    configfile << metadata.byteArraySize << '\n';
    configfile << metadata.max_ngram_order << '\n';
    configfile << metadata.api_version << '\n';
    configfile << metadata.btree_node_size << '\n';
    configfile.close();
}

template<class StringType>
LM_metadata readConfigFile(const StringType path) {
    LM_metadata ret;
    std::ifstream configfile(path);

    if (configfile.fail()) {
        std::cerr << "Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;

    //Get byte array size
    getline(configfile, line);
    ret.byteArraySize = std::stoull(line.c_str());

    //Get max ngram order
    getline(configfile, line);
    ret.max_ngram_order = atoi(line.c_str());

    //Check api version
    getline(configfile, line);
    if (atof(line.c_str()) != API_VERSION) {
        std::cerr << "The gLM API has changed, please rebinarize your language model." << std::endl;
        exit(EXIT_FAILURE);
    }

    //Get btree_node_size
    getline(configfile, line);
    ret.btree_node_size = atoi(line.c_str());
}
