#include "trie.hh"

int main(int argc, char* argv[]) {
    ArpaReader pesho(argv[1]);
    processed_line text = pesho.readline();
    B_tree * root_btree = new B_tree(atoi(argv[2]));

    while (!text.filefinished){
        addToTrie(root_btree, text, pesho.max_ngrams, atoi(argv[2]));
        text = pesho.readline();
    }
    std::cout << "Finished constructing the trie!" << std::endl;
    //Compress the trie
    compressTrie(root_btree);

    std::cout << "Total trie array size is: " << calculateTrieSize(root_btree)/(1024*1024) << "MB." << std::endl;

    ArpaReader pesho2(argv[1]);
    processed_line text2 = pesho2.readline();

    while (!text2.filefinished) {
        std::pair<Entry, unsigned short> found = findNgram(root_btree, text2.ngrams);
        bool correct = false;
        if (found.second) {
            correct = found.first.prob == text2.score;
            correct = correct && (found.first.backoff == text2.backoff);
        } else {
            std::cout << "Ngram not found! " << std::endl;
        }
        if (!correct) {
            std::cout << text2 << std::endl;
            std::cout << "Ngram size is: " << text2.ngram_size << std::endl;
            std::cout << "There has been an error! Score: Expected " << text2.score
            << " Got: " << found.first.prob << " Backoff expected: " << text2.backoff << " Got: " << found.first.backoff << std::endl;
        }
        text2 = pesho2.readline();
    }

    //Free all the used memory
    deleteTrie(root_btree);
} 
