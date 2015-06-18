#include <vector>
#include <string.h>
#include <limits>
#include <exception>

class offsetTooBig: public std::exception
{
  virtual const char* what() const throw()
  {
    return "The offset is too big to fit in unsigned int. The trie needs to be sharded.";
  }
} offsetEx;

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

unsigned char getEntrySize(bool pointer2Index = false) {
    /*This function returns the size of all individual components of the struct.
    It is necessary for bit packing, because we don't want to store padding*/
    if (pointer2Index) { //Use the pointer as index offset
        return sizeof(unsigned int) + sizeof(unsigned int) + 2*sizeof(float);
    } else {
        return sizeof(unsigned int) + sizeof(B_tree *) + 2*sizeof(float);
    }
    
}

void EntryToByteArray(std::vector<unsigned char> &byte_array, Entry& entry, bool pointer2Index = false) noexcept(false) {
    /*Converts an Entry to a byte array and appends it to the given vector of bytes*/
    unsigned char entry_size_bytes = getEntrySize(pointer2Index);
    unsigned char temparr[entry_size_bytes]; //Array used as a temporary container
    unsigned char accumulated_size = 0;  //How much we have copied thus far

    memcpy(&temparr[accumulated_size], (unsigned char *)&entry.value, sizeof(entry.value));
    accumulated_size+= sizeof(entry.value);

    //Convert the next_level to bytes. It could be a pointer or an unsigned int
    if (pointer2Index) {
        if (std::numeric_limits<unsigned int>::max() < (size_t)entry.next_level) {
            throw offsetEx;
        }
        unsigned char next_level_size = sizeof(unsigned int);
        memcpy(&temparr[accumulated_size], (unsigned char *)&entry.next_level, next_level_size);
        accumulated_size+=next_level_size;
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

    unsigned char accumulated_size = 0; //Keep track of the array index

    memcpy((unsigned char *)&value, &input_arr[accumulated_size], sizeof(value));
    accumulated_size+= sizeof(value);

    //If we have a offset instead of pointer we read in less bytes (4 vs 8)
    if (pointer2Index) {
        next_level = 0; //Excplicitly set all bits to 0 otherwise it will ruin our conversion since we are
                        //only writing to half of the bytes.
        unsigned char next_level_size = sizeof(unsigned int);
        memcpy((unsigned char *)&next_level, &input_arr[accumulated_size], next_level_size);
        accumulated_size+=next_level_size;
    } else {
        memcpy((unsigned char *)&next_level, &input_arr[accumulated_size], sizeof(next_level));
        accumulated_size+=sizeof(next_level);
    }
    
    memcpy((unsigned char *)&prob, &input_arr[accumulated_size], sizeof(prob));
    accumulated_size+=sizeof(prob);

    memcpy((unsigned char *)&backoff, &input_arr[accumulated_size], sizeof(backoff));

    Entry ret = {value, next_level, prob, backoff};
    return ret;
}
