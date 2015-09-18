#include "btree.hh"
#include <stdlib.h>

int main(int argc, char* argv[]) {
    const char * filename1 = "/tmp/graph.dot";
    const char * filename2 = "/tmp/graph_compressed.dot";

    if (argc == 3) {
        filename1 = argv[3];
        filename2 = argv[4];
    }
    const int max_degree = 7;
    const int num_entries = 24;

    unsigned int numbers[num_entries] = {10, 12, 13, 15, 17, 19, 20, 24, 27, 35, 41, 45, 47, 53, 56, 57, 60, 61, 63, 66, 67, 74, 78, 81};

    B_tree * pesho = new B_tree(max_degree);
    for (auto num : numbers) {
        Entry new_entry_actual = {num, nullptr, 0.0, 0.0};
        pesho->insert_entry(new_entry_actual);
    }

    pesho->produce_graph(filename1);

    pesho->compress();
    pesho->produce_graph(filename2);
    delete pesho;

    return 0;

}