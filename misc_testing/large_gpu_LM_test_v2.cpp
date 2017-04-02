 #include "search_utilities_v2.hh"

int main(int argc, char* argv[]) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " path_to_arpafile path_to_binary_model flush_value [deviceID=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int deviceID = 0;
    if (argc == 5) {
        deviceID = atoi(argv[4]);
    }
    
    LM lm(argv[2]);

    GPUSearcher engine(1, lm, deviceID);

    ArpaReader pesho2(argv[1]);
    processed_line text2 = pesho2.readline();

    std::vector<unsigned int> keys;
    std::vector<float> check_against;

    while (!text2.filefinished) {
        unsigned int num_keys = 0; //How many ngrams are we going to query
        //Inefficient reallocation of keys_to_query. Should be done better
        int num_padded =  lm.metadata.max_ngram_order - text2.ngrams.size();
        for (int i = 0; i < num_padded; i++) {
            text2.ngrams.push_back(0); //Extend ngrams to max num ngrams if they are of lower order
        }
        
        for (unsigned int i = 0; i < lm.metadata.max_ngram_order; i++) {
            keys.push_back(text2.ngrams[i]); //Extend ngrams to max num ngrams if they are of lower order
        }

        check_against.push_back(text2.score);

        num_keys++;
        text2 = pesho2.readline();

        if (num_keys > atoi(argv[3])) {
            //Flush
            std::vector<float> results = engine.search(keys, 0);
            for (size_t i = 0; i < check_against.size(); i++) {
                if (check_against[i] != results[i]) {
                    std::cerr << "Error! Expected: " << check_against[i] << ", got: " << results[i] << std::endl;
                    std::cerr << "Problematic ngram is: ";
                    for (size_t j = i*lm.metadata.max_ngram_order; j < (i+1)*lm.metadata.max_ngram_order; j++) {
                        std::cerr << keys[j] << " ";
                    }
                    std::cerr << std::endl;
                }
            }
            keys.clear();
            check_against.clear();
        }
    }

    //Last bit
    std::vector<float> results = engine.search(keys, 0);
    for (size_t i = 0; i < check_against.size(); i++) {
        if (check_against[i] != results[i]) {
            std::cerr << "Error! Expected: " << check_against[i] << ", got: " << results[i] << std::endl;
            std::cerr << "Problematic ngram is: ";
            for (size_t j = i*lm.metadata.max_ngram_order; j < (i+1)*lm.metadata.max_ngram_order; j++) {
                std::cerr << keys[j] << " ";
            }
            std::cerr << std::endl;
        }
    }
    return 0;
}
