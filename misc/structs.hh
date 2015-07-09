#include <vector>
#include <map>
#include <string.h>
#include <iostream>
#include "entry_structs.hh"
#define API_VERSION 1.0
#pragma once

//Metadata to write on a config file.
struct LM_metadata {
    size_t byteArraySize;  //Size in bytes
    unsigned short max_ngram_order;
    float api_version;
    unsigned short btree_node_size;
};

bool operator== (const LM_metadata &left, const LM_metadata &right) {
    return left.byteArraySize == right.byteArraySize && left.max_ngram_order == right.max_ngram_order &&
           left.api_version == right.api_version && left.btree_node_size == right.btree_node_size;
}

std::ostream& operator<< (std::ostream &out, LM_metadata &metadata) {
    out << "Api version: " << metadata.api_version << std::endl
    << "Byte array size: " << metadata.byteArraySize << std::endl
    << "Size of the datasctructure in memory is: " << metadata.byteArraySize/(1024*1024) << " MB."<< std::endl
    << "Max ngram order: " << metadata.max_ngram_order << std::endl;
    return out;
}

//A struct that contains all possible and necessary information for an LM
struct LM {
    std::vector<unsigned char> trieByteArray;
    std::map<std::string, unsigned int> encode_map;
    std::map<unsigned int, std::string> decode_map;
    LM_metadata metadata;
};

void EntryToByteArray(std::vector<unsigned char> &byte_array, Entry& entry, bool pointer2Index = false) {
    /*Converts an Entry to a byte array and appends it to the given vector of bytes*/
    unsigned char entry_size_bytes = getEntrySize(pointer2Index);
    unsigned char temparr[entry_size_bytes]; //Array used as a temporary container
    unsigned char accumulated_size = 0;  //How much we have copied thus far

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.value, sizeof(entry.value));
    accumulated_size+= sizeof(entry.value);

    //Convert the next_level to bytes. It could be a pointer or an unsigned int
    if (pointer2Index) {
        memcpy(&temparr[accumulated_size], (unsigned char *)&entry.offset, sizeof(entry.offset));
        accumulated_size+=sizeof(entry.offset);
    } else {
        memcpy(&temparr[accumulated_size], (unsigned char *)&entry.next_level, sizeof(entry.next_level));
        accumulated_size+=sizeof(entry.next_level);
    }

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.prob, sizeof(entry.prob));
    accumulated_size+=sizeof(entry.prob);

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.backoff, sizeof(entry.backoff));

    for (unsigned char i = 0; i < entry_size_bytes; i++) {
        byte_array.push_back(temparr[i]);
    }
}

Entry byteArrayToEntry(unsigned char * input_arr, bool pointer2Index = false) {
    //MAKE SURE YOU FREE THE ARRAY!
    unsigned int value;
    B_tree * next_level;
    float prob;
    float backoff;
    unsigned int offset;

    unsigned char accumulated_size = 0; //Keep track of the array index

    memcpy((unsigned char *)&value, &input_arr[accumulated_size], sizeof(value));
    accumulated_size+= sizeof(value);

    //If we have a offset instead of pointer we read in less bytes (4 vs 8)
    if (pointer2Index) {
        next_level = nullptr; //we only have information about the offset here so the pointer is invalid;

        memcpy((unsigned char *)&offset, &input_arr[accumulated_size], sizeof(offset));
        accumulated_size+=sizeof(offset);
    } else {
        offset = 0; //Default offset. We only set it if pointer2Index is true;

        memcpy((unsigned char *)&next_level, &input_arr[accumulated_size], sizeof(next_level));
        accumulated_size+=sizeof(next_level);
    }
    
    memcpy((unsigned char *)&prob, &input_arr[accumulated_size], sizeof(prob));
    accumulated_size+=sizeof(prob);

    memcpy((unsigned char *)&backoff, &input_arr[accumulated_size], sizeof(backoff));

    Entry ret = {value, next_level, prob, backoff, offset};
    return ret;
}
