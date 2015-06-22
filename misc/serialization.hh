#include <iostream>
#include <fstream>
#include <string>
#include "structs.hh"

//Use boost to serialize the vectors
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/filesystem.hpp>

//Compress the binary format to bzip2
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp> 

template<class StringType, class DataStructure>
void serializeDatastructure(DataStructure& byte_arr, const StringType path){
    std::ofstream os (path, std::ios::binary);

    {// The boost archive needs to be in a separate scope otherwise it doesn't flush properly
        //Create a bzip2 compressed filtered stream
        boost::iostreams::filtering_stream<boost::iostreams::output> filtered_stream;
        filtered_stream.push(boost::iostreams::bzip2_compressor());
        filtered_stream.push(os);

        boost::archive::text_oarchive oarch(filtered_stream);
        oarch << byte_arr;
    }
    os.close();
}

//The byte_arr vector should be reserved to prevent constant realocation.
template<class StringType, class DataStructure>
void readDatastructure(DataStructure& byte_arr, const StringType path){
    std::ifstream is (path, std::ios::binary);

    if (is.fail() ){
        std::cerr << "Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    {// The boost archive needs to be in a separate scope otherwise it doesn't flush properly
        //Create a bzip2 compressed filtered stream for decopression
        boost::iostreams::filtering_stream<boost::iostreams::input> filtered_stream;
        filtered_stream.push(boost::iostreams::bzip2_decompressor());
        filtered_stream.push(is);

        boost::archive::text_iarchive iarch(filtered_stream);
        iarch >> byte_arr;
    }
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
    //Also store in the config file the size of the datastructure. Useful to know if we can fit our model
    //on the available GPU memory, but we don't actually need to ever read it back. It is for the user's benefit.
    configfile << "BTree Trie memory size: " << metadata.byteArraySize/(1024*1024) << " MB\n";
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

//Given a path and an LM binarizes the LM there
template<class StringType>
void writeBinary(const StringType path, LM& lm) {
    createDirIfnotPresent(path);
    std::string basepath(path);
    storeConfigFile(lm.metadata, basepath + "/config");
    serializeDatastructure(lm.trieByteArray, basepath + "/lm.bin");
    serializeDatastructure(lm.encode_map, basepath + "/encode.map");
    serializeDatastructure(lm.decode_map, basepath + "/decode.map");
}

//Reads the model into the given (presumably empty byte_arr)
template<class StringType>
void readBinary(const StringType path, LM& lm) {
    std::string basepath(path);
    lm.metadata = readConfigFile(basepath + "/config");
    lm.trieByteArray.reserve(lm.metadata.byteArraySize);
    readDatastructure(lm.trieByteArray, basepath + "/lm.bin");
    readDatastructure(lm.encode_map, basepath + "/encode.map");
    readDatastructure(lm.decode_map, basepath + "/decode.map");
}
