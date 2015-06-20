#include <iostream>
#include <fstream>
#include <string>
#include "structs.hh"

//Use boost to serialize the vectors
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>

#define API_VERSION 1.0

template<class StringType>
void serializeByteArray(std::vector<unsigned char>& byte_arr, const StringType path){
    std::ofstream os (path, std::ios::binary);
    boost::archive::text_oarchive oarch(os);
    oarch << byte_arr;
    os.close();
}

//The byte_arr vector should be reserved to prevent constant realocation.
template<class StringType>
void readByteArray(std::vector<unsigned char>& byte_arr, const StringType path){
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
    } else {
        ret.api_version = atof(line.c_str());
    }

    //Get btree_node_size
    getline(configfile, line);
    ret.btree_node_size = atoi(line.c_str());
    return ret;
}

template<class StringType>
void createDirIfnotPresent(const StringType path) {
    boost::filesystem::path dir(path);

    if (!(boost::filesystem::exists(dir))) {
        //Directory doesn't exist, try to create it
        if (boost::filesystem::create_directory(dir)) {
            //all is good!
        } else {
            std::cerr << "Failed to create a directory at " << path << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

//Given a path to a btree_trie byte array and metadata, stores the model
template<class StringType>
void writeBinary(const StringType path, std::vector<unsigned char>& byte_arr, LM_metadata metadata) {
    createDirIfnotPresent(path);
    std::string basepath(path);
    storeConfigFile(metadata, basepath + "/config");
    serializeByteArray(byte_arr, basepath + "/lm.bin");
}

//Reads the model into the given (presumably empty byte_arr)
template<class StringType>
LM_metadata readBinary(const StringType path, std::vector<unsigned char>& byte_arr) {
    std::string basepath(path);
    LM_metadata metadata = readConfigFile(basepath + "/config");
    byte_arr.reserve(metadata.byteArraySize);
    readByteArray(byte_arr, basepath + "/lm.bin");
    return metadata;
}
