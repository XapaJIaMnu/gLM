#include "tokenizer.hh"

int main(int argc, char* argv[]) {
    ArpaReader pesho(argv[1]);
    processed_line text = pesho.readline();
    while (!text.filefinished){
        std::cout << text << std::endl;
        text = pesho.readline();
    }
}