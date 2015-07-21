#include "btree.hh"
#include "memory_management.hh"
#include "gpu_search.hh"

int main(int argc, char* argv[]) {
    //Defaults
    int max_degree = 5;
    unsigned int num_entries = 125;
    const char * filename = "/tmp/graph_compressed.dot";

    if (argc == 4) {
        max_degree = atoi(argv[1]);
        num_entries = atoi(argv[2]);
        filename = argv[3];
    }

    B_tree * pesho = new B_tree(max_degree);
    std::set<unsigned int> prev_nums; //Used to see if we have duplicating nums
    while (prev_nums.size() < num_entries) {
        unsigned int new_entry = rand() % (num_entries*10);
        if (prev_nums.count(new_entry) == 0){
            Entry new_entry_actual = {new_entry, nullptr, 0.5, 0.75};
            pesho->insert_entry(new_entry_actual);
            prev_nums.insert(new_entry);
        }
    }
    
    //Compres the btree
    pesho->compress();
    pesho->produce_graph(filename);

    //produce btree array
    std::vector<unsigned char> byte_arr;
    byte_arr.reserve(pesho->getTotalTreeSize());
    pesho->toByteArray(byte_arr, true /*pointer2index*/);

    std::pair<bool, std::string> test_res2 = test_btree_array(prev_nums, byte_arr, max_degree, true);
    if (!test_res2.first) {
        std::cout << test_res2.second << std::endl;
    }

    unsigned char * gpuByteArray = copyToGPUMemory(byte_arr.data(), byte_arr.size());
    for (std::set<unsigned int>::iterator it = prev_nums.begin(); it != prev_nums.end(); it++){
        unsigned short * first_node_size = reinterpret_cast<unsigned short *>(&byte_arr.data()[0]);
        //std::cout << "Value at: " << *it << std::endl;
        if (*it == 857) {
            searchWrapper(gpuByteArray, 0, *first_node_size, *it, 1, 5);
            cudaDevSync();
        }
    }
    freeGPUMemory(gpuByteArray);

}
