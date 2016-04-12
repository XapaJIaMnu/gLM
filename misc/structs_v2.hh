struct Entry_v2 {
    unsigned int vocabID;
    float prob;
    float backoff;
};

#ifndef   COMPARISON_OVERLOADS
#define   COMPARISON_OVERLOADS

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
#endif
