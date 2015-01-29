#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

struct Entry {
    unsigned int value;
    Entry * next_level;
    bool last;
};


__device__ int binary_search(Entry * current_level, unsigned int word, unsigned int length, unsigned int curent) {

    for (int i 0; i < )
    if current_level[current].value < word

}

int main(){

    thrust::device_vector<Entry> lvl1(3);
    Entry entry1 = {3, NULL, true};
    Entry entry2 = {4, NULL, true};
    Entry entry3 = {5, NULL, true};
    lvl1[0] = entry1;
    lvl1[1] = entry2;
    lvl1[1] = entry3;

    Entry tmpentry = {3, NULL, true};
    Entry tmpentry1 = {4, NULL, true};
    Entry tmpentry2 = {0, NULL, true};

    //std::cout << thrust::binary_search(lvl1.begin(), lvl1.end(), tmpentry) << std::endl;
    //std::cout << thrust::binary_search(lvl1.begin(), lvl1.end(), tmpentry1) << std::endl;
    //std::cout << thrust::binary_search(lvl1.begin(), lvl1.end(), tmpentry2) << std::endl;

    return 0;
}
