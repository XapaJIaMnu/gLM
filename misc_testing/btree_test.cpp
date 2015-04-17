#include "btree.hh"
#include <stdlib.h>
#include <set>

int main(int argc, char* argv[]) {
    int max_degree = 3;
    const char * filename1 = "graph.dot";
    const char * filename2 = "graph_compressed.dot";
    int num_entries = 25;

    if (argc == 5) {
        max_degree = atoi(argv[1]);
        num_entries = atoi(argv[2]);
        filename1 = argv[3];
        filename2 = argv[4];
    }
    B_tree * pesho = new B_tree(max_degree);
    std::set<unsigned int> prev_nums; //Used to see if we have duplicating nums
    while (prev_nums.size() < num_entries) {
        unsigned int new_entry = rand() % (num_entries*10);
        if (prev_nums.count(new_entry) == 0){
            Entry new_entry_actual = {new_entry, nullptr, false};
            pesho->insert_entry(new_entry_actual);
            prev_nums.insert(new_entry);
        }
    }

    pesho->produce_graph(filename1);
    test_btree(prev_nums, pesho);

    pesho->compress();
    pesho->produce_graph(filename2);
    //test_btree(prev_nums, pesho);

    delete pesho;
    return 0;
}