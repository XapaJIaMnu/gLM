#define API_VERSION 1.6
#include <iostream>
#include <fstream>
#include <string>
//Since boost compressed IO streams are SLOW as hell, use mmap'd IO
#include <sys/mman.h>
#include <fcntl.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#pragma once

//Metadata to write on a config file.
struct LM_metadata {
    size_t byteArraySize;  //Size in bytes
    unsigned short max_ngram_order;
    float api_version;
    unsigned short btree_node_size;
    bool mmapd = false; //Identify whether we have the memory mapped version or not. False by default so we don't get a segfault in destructor.
};

inline bool operator== (const LM_metadata &left, const LM_metadata &right) {
    return left.byteArraySize == right.byteArraySize && left.max_ngram_order == right.max_ngram_order &&
           left.api_version == right.api_version && left.btree_node_size == right.btree_node_size;
};

inline std::ostream& operator<< (std::ostream &out, LM_metadata &metadata) {
    out << "Api version: " << metadata.api_version << std::endl
    << "Byte array size: " << metadata.byteArraySize << std::endl
    << "Size of the datasctructure in memory is: " << metadata.byteArraySize/(1024*1024) << " MB."<< std::endl
    << "Btree node size is: " << metadata.btree_node_size << std::endl
    << "Max ngram order: " << metadata.max_ngram_order << std::endl;
    return out;
};

//A struct that contains all possible and necessary information for an LM
class LM {
    private:
        template<class StringType>
        void readConfigFile(const StringType path);
        template<class StringType>
        void storeConfigFile(const StringType path, bool compactStorage);
        unsigned char * mmapedByteArray;
    public:
        std::vector<unsigned char> trieByteArray;
        std::unordered_map<std::string, unsigned int> encode_map;
        std::unordered_map<unsigned int, std::string> decode_map;
        LM_metadata metadata;

        //Constructors:
        template<class StringType> 
        LM(StringType); //Create an LM from serialized files on disk
        LM(){}; //Create LM object to populate during construction. Use default construtor

        //Destructor
        //A destructor. We need to check wheter the memory map needs to be undone
        ~LM() {
            if (metadata.mmapd) {
                munmap(reinterpret_cast<void *>(mmapedByteArray), metadata.byteArraySize);
            }
        }

        //Write to disk:
        template<class StringType> 
        void writeBinary(const StringType path, bool compactStorage = false);
};


unsigned char * readMmapTrie(const char * filename, size_t size);

template<class StringType>
void createDirIfnotPresent(const StringType path);

template<class StringType, class DataStructure>
void readDatastructure(DataStructure& byte_arr, const StringType path);

template<class StringType, class DataStructure>
void serializeDatastructure(DataStructure& byte_arr, const StringType path);
