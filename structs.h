struct Entry {
    unsigned int value;
    Entry * next_level;
    bool last;
};

bool operator> (const Entry &left, const Entry &right) {
    return (left.value > right.value);
}

bool operator< (const Entry &left, const Entry &right) {
    return (left.value < right.value);
}

bool operator== (const Entry &left, const Entry &right) {
    return (left.value == right.value);
}

bool operator!= (const Entry &left, const Entry &right) {
    return (left.value != right.value);
}

//With a number
bool operator> (const unsigned int &left, const Entry &right) {
    return (left > right.value);
}

bool operator< (const unsigned int &left, const Entry &right) {
    return (left < right.value);
}

bool operator== (const unsigned int &left, const Entry &right) {
    return (left == right.value);
}

bool operator!= (const unsigned int &left, const Entry &right) {
    return (left != right.value);
}

//The other way, for convenience

bool operator> (const Entry &left, const unsigned int &right) {
    return (left.value > right);
}

bool operator< (const Entry &left, const unsigned int &right) {
    return (left.value < right);
}

bool operator== (const Entry &left, const unsigned int &right) {
    return (left.value == right);
}

bool operator!= (const Entry &left, const unsigned int &right) {
    return (left.value != right);
}