#include <vector>
#include <map>
#include <cstring>
#include <iostream>
#include "entry_structs.hh"
#define API_VERSION 1.1
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

void EntriesToByteArray(std::vector<unsigned char> &byte_array, std::vector<Entry> &entries, bool pointer2Index = false) {
    //Appends a multitude of entries to the byte array.
    //Entries are stored in the following way: ALLKEYS_datakey1_datakey2_etc
    //This is a bit counter intuitive but permits for better memory accesses on the gpu and is similar to a b+tree

    unsigned int offset = byte_array.size(); //Keep track of where in the byte array we start putting things

    //First push the values onto the byte array
    //We use resize a bit hacky here - We are changing the .size() value so that when we use push_back it works
    //Basically we bump the size for all elements we are going to insert using memcpy.
    byte_array.resize(byte_array.size() + getEntrySize(pointer2Index)*entries.size());

    for (auto entry : entries) {
        std::memcpy(byte_array.data() + offset, &entry.value, sizeof(entry.value));
        offset += sizeof(entry.value);
    }

    //Then everything else
    if (pointer2Index) {
        for (auto entry : entries) {
            std::memcpy(byte_array.data() + offset, &entry.offset, sizeof(entry.offset));
            offset += sizeof(entry.offset);
            std::memcpy(byte_array.data() + offset, &entry.prob, sizeof(entry.prob));
            offset += sizeof(entry.prob);
            std::memcpy(byte_array.data() + offset, &entry.backoff, sizeof(entry.backoff));
            offset += sizeof(entry.backoff);
        }
    } else {
        for (auto entry : entries) {
            std::memcpy(byte_array.data() + offset, &entry.next_level, sizeof(entry.next_level));
            offset += sizeof(entry.next_level);
            std::memcpy(byte_array.data() + offset, &entry.prob, sizeof(entry.prob));
            offset += sizeof(entry.prob);
            std::memcpy(byte_array.data() + offset, &entry.backoff, sizeof(entry.backoff));
            offset += sizeof(entry.backoff);
        }
    }
}

Entry * byteArrayToEntries(std::vector<unsigned char> &byte_array, int num_entries, unsigned int start_idx, bool pointer2Index = false) {
    //We return a pointer to array of entries. Burden of free is on the calling function!!!
    Entry * entries = new Entry[num_entries];
    unsigned int current_idx = start_idx;

    //First get all values
    for (int i = 0; i < num_entries; i++) {
        std::memcpy(&entries[i].value, &byte_array[current_idx], sizeof(entries[i].value));
        current_idx += sizeof(entries[i].value);
    }

    //Then the rest of the information from the entries
    if (pointer2Index) {
        for (int i = 0; i < num_entries; i++) {
            std::memcpy(&entries[i].offset, &byte_array[current_idx], sizeof(entries[i].offset));
            current_idx += sizeof(entries[i].offset);
            std::memcpy(&entries[i].prob, &byte_array[current_idx], sizeof(entries[i].prob));
            current_idx += sizeof(entries[i].prob);
            std::memcpy(&entries[i].backoff, &byte_array[current_idx], sizeof(entries[i].backoff));
            current_idx += sizeof(entries[i].backoff);

            //Set the unused B_tree * to nullptr to catch errors if somebody tries to use it
            entries[i].next_level = nullptr;
        }
    } else {
        for (int i = 0; i < num_entries; i++) {
            std::memcpy(&entries[i].next_level, &byte_array[current_idx], sizeof(entries[i].next_level));
            current_idx += sizeof(entries[i].next_level);
            std::memcpy(&entries[i].prob, &byte_array[current_idx], sizeof(entries[i].prob));
            current_idx += sizeof(entries[i].prob);
            std::memcpy(&entries[i].backoff, &byte_array[current_idx], sizeof(entries[i].backoff));
            current_idx += sizeof(entries[i].backoff);

            //Set the unused offset to 0.
            entries[i].offset = 0;
        }
    }

    return entries;
}
