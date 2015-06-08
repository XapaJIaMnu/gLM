#include <vector>
#include <string.h>

class B_tree; //Forward declaration

struct Entry {
    unsigned int value;
    B_tree * next_level;
    float prob;
    float backoff;
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

unsigned char getEntrySize() {
    /*This function returns the size of all individual components of the struct.
    It is necessary for bit packing, because we don't want to store padding*/
    return sizeof(unsigned int) + sizeof(B_tree *) + 2*sizeof(float);
}

void EntryToByteArray(std::vector<unsigned char> &byte_array, Entry& entry){
    /*Converts an Entry to a byte array and appends it to the given vector of bytes*/
    unsigned char entry_size_bytes = getEntrySize();
    unsigned char temparr[entry_size_bytes]; //Array used as a temporary container
    unsigned char accumulated_size = 0;  //How much we have copied thus far

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.value, sizeof(entry.value));
    accumulated_size+= sizeof(entry.value);

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.next_level, sizeof(entry.next_level));
    accumulated_size+=sizeof(entry.next_level);

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.prob, sizeof(entry.prob));
    accumulated_size+=sizeof(entry.prob);

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.backoff, sizeof(entry.backoff));

    for (unsigned char i = 0; i < entry_size_bytes; i++) {
        byte_array.push_back(temparr[i]);
    }
}

Entry byteArrayToEntry(unsigned char * input_arr) {
    //MAKE SURE YOU FREE THE ARRAY!
    unsigned int value;
    B_tree * next_level;
    float prob;
    float backoff;

    unsigned char accumulated_size = 0; //Keep track of the array index

    memcpy((unsigned char *)&value, &input_arr[accumulated_size], sizeof(value));
    accumulated_size+= sizeof(value);

    memcpy((unsigned char *)&next_level, &input_arr[accumulated_size], sizeof(next_level));
    accumulated_size+=sizeof(next_level);

    memcpy((unsigned char *)&prob, &input_arr[accumulated_size], sizeof(prob));
    accumulated_size+=sizeof(prob);

    memcpy((unsigned char *)&backoff, &input_arr[accumulated_size], sizeof(backoff));

    Entry ret = {value, next_level, prob, backoff};
    return ret;
}
