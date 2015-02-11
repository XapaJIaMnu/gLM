#include "b_treev2.hh"

int main() {
    B_tree * pesho = new B_tree(3);
    pesho->insert_entry(3);
    pesho->insert_entry(6);
    pesho->insert_entry(7);
    pesho->insert_entry(9);
    pesho->insert_entry(2);
    pesho->insert_entry(17);
    pesho->insert_entry(23);
    pesho->insert_entry(25);
    pesho->insert_entry(26);
    pesho->insert_entry(27);
    pesho->insert_entry(28);
    pesho->insert_entry(29);
    pesho->insert_entry(35);
    pesho->insert_entry(37);
    /*pesho->insert_entry(40);
    pesho->insert_entry(51);
    pesho->insert_entry(63);
    pesho->insert_entry(1);
    pesho->insert_entry(11);
    pesho->insert_entry(15);
    pesho->insert_entry(19);*/
    pesho->draw_tree();
    delete pesho;
    return 0;
}