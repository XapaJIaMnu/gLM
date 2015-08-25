#include "btree.hh"
#include "memory_management.hh"
#include "gpu_search.hh"
#include <chrono>
#include <ctime>

int main(int argc, char* argv[]) {
    //Time events:
    std::chrono::time_point<std::chrono::system_clock> start, all_btree_done, cpuBtreeTest, memcpyBytearray, memcpyKeysStart, memcpyKeysEnd, kernel, memFree;
    start = std::chrono::system_clock::now();

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
            Entry new_entry_actual = {new_entry, nullptr, 0.5f + new_entry, 0.75f + new_entry};
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

    all_btree_done = std::chrono::system_clock::now();

    std::pair<bool, std::string> test_res2 = test_btree_array(prev_nums, byte_arr, max_degree, true);
    if (!test_res2.first) {
        std::cout << test_res2.second << std::endl;
    }

    cpuBtreeTest = std::chrono::system_clock::now();

    unsigned char * gpuByteArray = copyToGPUMemory(byte_arr.data(), byte_arr.size());

    memcpyBytearray = std::chrono::system_clock::now();

    std::vector<unsigned int> keys_cpu;

    for (std::set<unsigned int>::iterator it = prev_nums.begin(); it != prev_nums.end(); it++){
        keys_cpu.push_back(*it);
    }
    keys_cpu.push_back(2341);

    memcpyKeysStart = std::chrono::system_clock::now();
    unsigned int * keys_gpu = copyToGPUMemory(keys_cpu.data(), keys_cpu.size());
    memcpyKeysEnd = std::chrono::system_clock::now();

    searchWrapper(gpuByteArray, 0, keys_gpu, keys_cpu.size()); //Test key not found
    cudaDevSync();
    kernel = std::chrono::system_clock::now();

    freeGPUMemory(gpuByteArray);
    freeGPUMemory(keys_gpu);

    memFree = std::chrono::system_clock::now();

    std::chrono::duration<double> diff = all_btree_done - start;
    std::cout << "Btree built for: " << diff.count() << " seconds" << std::endl;
    
    diff = cpuBtreeTest - all_btree_done;
    std::cout << "Btree cpu test: " << diff.count() << " seconds" << std::endl;
    
    diff = memcpyBytearray - cpuBtreeTest;
    std::cout << "Btree gpu copy: " << diff.count() << " seconds" << std::endl;
    
    diff = memcpyKeysEnd - memcpyKeysStart;
    std::cout << "Keys GPU copy: " << diff.count() << " seconds" << std::endl;
    
    diff = kernel - memcpyKeysEnd;
    std::cout << "Kernel execution:  " << diff.count() << " seconds" << std::endl;
    
    diff = memFree - kernel;
    std::cout << "Memory free time: " << diff.count() << " seconds" << std::endl;
}
