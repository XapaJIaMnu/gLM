#define API_VERSION 2.2
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
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
    size_t intArraySize; //Size of the first level of the trie, which is an array
    unsigned short max_ngram_order;
    float api_version;
    unsigned short btree_node_size;
};

inline bool operator== (const LM_metadata &left, const LM_metadata &right) {
    return left.byteArraySize == right.byteArraySize && left.max_ngram_order == right.max_ngram_order &&
           left.api_version == right.api_version && left.btree_node_size == right.btree_node_size &&
           left.intArraySize == right.intArraySize;
};

inline std::ostream& operator<< (std::ostream &out, LM_metadata &metadata) {
    out << "Api version: " << metadata.api_version << std::endl
    << "Byte array size: " << metadata.byteArraySize << std::endl
    << "First_level size: " << metadata.intArraySize << std::endl
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
        void storeConfigFile(const StringType path);
        unsigned char * mmapedByteArray;
        unsigned int * mmapedFirst_lvl;
        bool diskIO = false; //Keep track whether we have a readIN binarized LM or an LM created in memory.
    public:
        std::vector<unsigned char> trieByteArray;
        std::vector<unsigned int> first_lvl;
        std::unordered_map<std::string, unsigned int> encode_map;
        std::unordered_map<unsigned int, std::string> decode_map;
        LM_metadata metadata;

        //Constructors:
        template<class StringType> 
        LM(StringType); //Create an LM from serialized files on disk
        LM(){}; //Create LM object to populate during construction. Use default construtor

        //Destructor. Undo memory maps
        ~LM() {
            if (diskIO) {
                munmap(reinterpret_cast<void *>(mmapedByteArray), metadata.byteArraySize);
                //To maintain compatibility with the old format for now, check if intArraySize is more than 0
                //before attempting to execute memory map
                if (metadata.intArraySize) {
                    munmap(reinterpret_cast<void *>(mmapedByteArray), metadata.intArraySize);
                }
            }
        }

        //Write to disk:
        template<class StringType> 
        void writeBinary(const StringType path);
};


void * readMmapTrie(const char * filename, size_t size);

template<class StringType>
void createDirIfnotPresent(const StringType path);

template<class StringType, class DataStructure>
void readDatastructure(DataStructure& byte_arr, const StringType path);

template<class StringType, class DataStructure>
void serializeDatastructure(DataStructure& byte_arr, const StringType path);
