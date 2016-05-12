#pragma once

class B_tree; //Forward declaration

struct Entry_v2 {
    unsigned int vocabID;
    float prob;
    float backoff;
};

struct Entry {
    unsigned int value;
    B_tree * next_level;
    float prob;
    float backoff;
    unsigned int offset; //This is only used when converting the trie to byte array. It is not saved in any of the other serializations.
    //As such we don't care about its value in any other context and we don't test for it!
};

struct Entry_offset {
    unsigned int value;
    unsigned int next_level; //This used to be the offset
    float prob;
    float backoff;
};

inline bool operator> (const Entry &left, const Entry &right) {
    return (left.value > right.value);
}

inline bool operator< (const Entry &left, const Entry &right) {
    return (left.value < right.value);
}

inline bool operator== (const Entry &left, const Entry &right) {
    return (left.value == right.value);
}

inline bool operator!= (const Entry &left, const Entry &right) {
    return (left.value != right.value);
}

//With a number
inline bool operator> (const unsigned int &left, const Entry &right) {
    return (left > right.value);
}

inline bool operator< (const unsigned int &left, const Entry &right) {
    return (left < right.value);
}

inline bool operator== (const unsigned int &left, const Entry &right) {
    return (left == right.value);
}

inline bool operator!= (const unsigned int &left, const Entry &right) {
    return (left != right.value);
}

//The other way, for convenience

inline bool operator> (const Entry &left, const unsigned int &right) {
    return (left.value > right);
}

inline bool operator< (const Entry &left, const unsigned int &right) {
    return (left.value < right);
}

inline bool operator== (const Entry &left, const unsigned int &right) {
    return (left.value == right);
}

inline bool operator!= (const Entry &left, const unsigned int &right) {
    return (left.value != right);
}

inline bool operator> (const Entry_v2 &left, const Entry_v2 &right) {
    return (left.vocabID > right.vocabID);
}

inline bool operator< (const Entry_v2 &left, const Entry_v2 &right) {
    return (left.vocabID < right.vocabID);
}

inline bool operator== (const Entry_v2 &left, const Entry_v2 &right) {
    return (left.vocabID == right.vocabID);
}

inline bool operator!= (const Entry_v2 &left, const Entry_v2 &right) {
    return (left.vocabID != right.vocabID);
}

//With a number
inline bool operator> (const unsigned int &left, const Entry_v2 &right) {
    return (left > right.vocabID);
}

inline bool operator< (const unsigned int &left, const Entry_v2 &right) {
    return (left < right.vocabID);
}

inline bool operator== (const unsigned int &left, const Entry_v2 &right) {
    return (left == right.vocabID);
}

inline bool operator!= (const unsigned int &left, const Entry_v2 &right) {
    return (left != right.vocabID);
}

//The other way, for convenience

inline bool operator> (const Entry_v2 &left, const unsigned int &right) {
    return (left.vocabID > right);
}

inline bool operator< (const Entry_v2 &left, const unsigned int &right) {
    return (left.vocabID < right);
}

inline bool operator== (const Entry_v2 &left, const unsigned int &right) {
    return (left.vocabID == right);
}

inline bool operator!= (const Entry_v2 &left, const unsigned int &right) {
    return (left.vocabID != right);
}


inline unsigned char getEntrySize(bool pointer2Index = false) {
    /*This function returns the size of all individual components of the struct.
    It is necessary because we store either the B_tree * next_level or the offset, never both*/
    if (pointer2Index) { //Use the pointer as index offset
        return sizeof(unsigned int) + sizeof(unsigned int) + 2*sizeof(float);
    } else {
        return sizeof(unsigned int) + sizeof(B_tree *) + 2*sizeof(float);
    }
    
}
