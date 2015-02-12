#include "b_treev2.hh"
#include <stdlib.h>
#include <set>

int main(int argc, char* argv[]) {
    int max_degree = 3;
    const char * filename = "graph.dot";
    int num_entries = 25;

    if (argc == 4) {
        max_degree = atoi(argv[1]);
        num_entries = atoi(argv[2]);
        filename = argv[3];
    }
    B_tree * pesho = new B_tree(max_degree);
    std::set<int> prev_nums; //Used to see if we have duplicating nums
    while (prev_nums.size() < num_entries) {
        int new_entry = rand() % (num_entries*10);
        if (prev_nums.count(new_entry) == 0){
            pesho->insert_entry(new_entry);
            prev_nums.insert(new_entry);
        }
    }

    pesho->produce_graph(filename);
    delete pesho;
    return 0;
}