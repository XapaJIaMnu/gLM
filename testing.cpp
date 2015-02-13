#include "structs.h"
#include <iostream>

int main() {
    Entry A = {10, nullptr, false};
    Entry B = {15, nullptr, false};
    Entry D = {16, nullptr, false};
    Entry E = {17, nullptr, false};
    Entry F = {18, nullptr, false};
    Entry G = {19, nullptr, false};
    Entry H = {22, nullptr, false};
    Entry I = {35, nullptr, false};

    std::cout << (A < B) << std::endl;
    std::cout << (E == D) << std::endl;
    std::cout << (10 == A) << std::endl;
    std::cout << (G == 22) << std::endl;
    std::cout << (B > 5) << std::endl;

}
