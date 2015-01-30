#include "b_tree.h"

int main() {
    B_tree * pesho = new B_tree(2);
    Entry A = {10, nullptr, false};
    Entry B = {15, nullptr, false};
    Entry D = {16, nullptr, false};
    Entry E = {17, nullptr, false};
    Entry F = {18, nullptr, false};
    Entry G = {19, nullptr, false};
    Entry H = {22, nullptr, false};
    Entry I = {35, nullptr, false};

    pesho->insert_Entry(A);
    pesho->insert_Entry(B);
    std::cout << "Here" << std::endl;
    pesho->insert_Entry(D);
    pesho->insert_Entry(E);
    std::cout << "Here too" << std::endl;
    pesho->insert_Entry(F);
    pesho->insert_Entry(G);
    pesho->insert_Entry(H);
    pesho->insert_Entry(I);
    //pesho->insert_Entry(A);
    pesho->draw_tree();
    delete pesho;
}
