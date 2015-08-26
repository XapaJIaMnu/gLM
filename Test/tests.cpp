#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Suites
//#define ARPA_TESTFILEPATH is defined by cmake
#define FLOAT_TOLERANCE 1e-5*1e-5
#include <boost/test/unit_test.hpp>
#include "trie.hh"  //Includes tokenizer.hh and btree.hh
#include "serialization.hh" //For testing properly serialization
#include <ctime>
#include <stdio.h>

//Float comparison
inline bool float_compare(float a, float b) { 

    return (a - b) * (a - b) < FLOAT_TOLERANCE;
}

//Init b_tree and the set of numbers that it contains
std::pair<B_tree *, std::set<unsigned int> > init_btree(int max_degree, unsigned int num_entries) {

    B_tree * pesho = new B_tree(max_degree);
    std::set<unsigned int> prev_nums; //Used to see if we have duplicating nums
    while (prev_nums.size() < num_entries) {
        unsigned int new_entry = 1 + (rand() % (num_entries*10));
        if (prev_nums.count(new_entry) == 0){
            Entry new_entry_actual = {new_entry, nullptr, 0.0, 0.0};
            pesho->insert_entry(new_entry_actual);
            prev_nums.insert(new_entry);
        }
    }
    return std::pair<B_tree *, std::set<unsigned int> >(pesho, prev_nums);
}

BOOST_AUTO_TEST_SUITE(Entry_overload)
BOOST_AUTO_TEST_CASE(various_tests) {
    Entry A = {10, nullptr, 0.0, 0.0};
    Entry B = {15, nullptr, 0.0, 0.0};
    Entry C = {10, nullptr, 0.0, 0.0};
    Entry D = {16, nullptr, 0.0, 0.0};
    Entry E = {17, nullptr, 0.0, 0.0};
    Entry G = {19, nullptr, 0.0, 0.0};

    BOOST_CHECK(A < B);
    BOOST_CHECK(A == C);
    BOOST_CHECK(E != D);
    BOOST_CHECK(10 == A);
    BOOST_CHECK(G != 22);
    BOOST_CHECK(B > 5);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(Btree)
 
BOOST_AUTO_TEST_CASE(Insert_in_order) {
    //Get the b_tree and the set
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(3, 25);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(pesho->root_node);

    int counter = 0;
    for (std::set<unsigned int>::iterator it = prev_nums.begin(); it != prev_nums.end(); it++) {
        BOOST_CHECK_MESSAGE( *it == iter->get_item(), "Expected: " << *it << ", got: " << iter->get_item() << " at position: " << counter << '.');
        counter++;
        iter->increment();
    }
    delete iter;
    delete pesho;

}

BOOST_AUTO_TEST_CASE(search_test) {
    //Get the b_tree and the set
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(3, 25);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    //Test search
    int position = 0;
    unsigned int value;
    std::set<unsigned int>::iterator it = prev_nums.begin();
    std::pair<B_tree_node *, int> res;
    std::pair<B_tree_node *, int> res2;

    //First case
    std::advance(it, position);
    Entry elem ={*it, nullptr, 0.0, 0.0};
    res = pesho->find_element(elem);
    value = res.first->words[res.second].value;
    BOOST_CHECK_MESSAGE(*it == value, "Expected: " << *it << ", got: " << value << " at position: " << position << "." );

    //Second case
    it = prev_nums.begin();
    position += 15;
    std::advance(it, position);
    Entry elem2 = {*it, nullptr, 0.0, 0.0};
    res2 = pesho->find_element(elem2);
    value = res2.first->words[res2.second].value;
    BOOST_CHECK_MESSAGE(*it == value, "Expected: " << *it << ", got: " << value << " at position: " << position << "." );

    delete pesho;
}

BOOST_AUTO_TEST_CASE(very_big_btree) {
    //init
    unsigned short max_btree_node_size = 17;
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(max_btree_node_size, 65000);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    //Check size:
    BOOST_CHECK_MESSAGE((pesho->size == 65000), "Wrong size! Got: " << pesho->size << " Expected: 65000.");

    //test
    std::pair<bool, std::string> test_result = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result.first, test_result.second);

    //Test compression:
    pesho->compress();
    std::pair<bool, std::string> test_result_compressed = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result_compressed.first, "Compressed: " << test_result_compressed.second);

    //Test conversion to byte array
    std::vector<unsigned char> byte_arr;
    size_t total_array_size = pesho->getTotalTreeSize();
    byte_arr.reserve(total_array_size);
    pesho->toByteArray(byte_arr);
    std::pair<bool, std::string> test_result_byte = test_btree_array(prev_nums, byte_arr, max_btree_node_size);
    BOOST_CHECK_MESSAGE(test_result_byte.first, "Byte array: " << test_result_byte.second);
    BOOST_CHECK_MESSAGE(total_array_size == byte_arr.size(), "Total size mismatch! Expected " << total_array_size << " actual " << byte_arr.size());

    delete pesho;
}

BOOST_AUTO_TEST_CASE(very_big_btree_small_node) {
    //init
    unsigned short max_btree_node_size = 5;
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(max_btree_node_size, 15000);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    //test
    std::pair<bool, std::string> test_result = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result.first, test_result.second);

    //Test compression:
    pesho->compress();
    std::pair<bool, std::string> test_result_compressed = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result_compressed.first, "Compressed: " << test_result_compressed.second);

    //Test conversion to byte array
    std::vector<unsigned char> byte_arr;
    size_t total_array_size = pesho->getTotalTreeSize();
    byte_arr.reserve(total_array_size);
    pesho->toByteArray(byte_arr);
    std::pair<bool, std::string> test_result_byte = test_btree_array(prev_nums, byte_arr, max_btree_node_size);
    BOOST_CHECK_MESSAGE(test_result_byte.first, "Byte array: " << test_result_byte.second);
    BOOST_CHECK_MESSAGE(total_array_size == byte_arr.size(), "Total size mismatch! Expected " << total_array_size << " actual " << byte_arr.size());

    delete pesho;
}

BOOST_AUTO_TEST_CASE(modify_entry_test) {
    //Get the b_tree and the set
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(3, 25);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    //Test search
    int position = 5;
    std::set<unsigned int>::iterator it = prev_nums.begin();
    std::advance(it, position);
    std::pair<B_tree_node *, int> res;
    std::pair<B_tree_node *, int> res2;


    Entry elem ={*it, nullptr, 0.0, 0.0};
    res = pesho->find_element(elem);

    //Change the value of the probability
    res.first->words[res.second].prob = 0.5;

    //Find the node again and check if the change was recorded
    Entry elem2 = {*it, nullptr, 0.0, 0.0};
    res2 = pesho->find_element(elem2);
    float current_prob = res2.first->words[res2.second].prob;
    BOOST_CHECK_MESSAGE(current_prob == 0.5, "Expected: " << 0.5 << ", got: " << current_prob << " at position: " << position << "." );
    delete pesho;
}
 
BOOST_AUTO_TEST_SUITE_END()
 
BOOST_AUTO_TEST_SUITE(Parser_and_trie)
 
BOOST_AUTO_TEST_CASE(max_ngrams_test) {
    ArpaReader pesho(ARPA_TESTFILEPATH);
    BOOST_CHECK_MESSAGE(pesho.max_ngrams == 5, "Wrong number of max ngrams. Got: " << pesho.max_ngrams << ", expected 5.");

    //Just iterate till the end of the file so that we don't have multiple file handles.
    processed_line text = pesho.readline();
    while (!text.filefinished){
        text = pesho.readline();
    }
}
 
BOOST_AUTO_TEST_CASE(random_arpa_lines_test) {
    ArpaReader pesho(ARPA_TESTFILEPATH);

    //Do several tests based on reading in the file
    for (int i = 0; i < 22404; i++) {
        processed_line text = pesho.readline();
        if (i == 4){
            BOOST_CHECK_MESSAGE(float_compare(text.backoff, -0.3436003), "Got " << text.backoff << " expected -0.3436003. Line number: " << i);
        }

        if (i == 12640){
            BOOST_CHECK_MESSAGE(float_compare(text.score, -1.0943657), "Got " << text.score << " expected -1.0943657. Line number: " << i);
            //Decode word
            std::map<unsigned int, std::string>::iterator it;
            std::string word = pesho.decode_map.find(text.ngrams[0])->second;
            BOOST_CHECK_MESSAGE(word == "having", "Got " << word << " expected having. Line number: " << i);
        }

        if (i == 15374){
            BOOST_CHECK_MESSAGE(float_compare(text.score, -0.83306277), "Got " << text.score << " expected -0.83306277. Line number: " << i);
            //Decode word
            std::map<unsigned int, std::string>::iterator it;
            std::string word = pesho.decode_map.find(text.ngrams[3])->second;
            BOOST_CHECK_MESSAGE(word == "safety", "Got " << word << " expected safety. Line number: " << i);
        }

        if (i == 22404){
            BOOST_CHECK_MESSAGE(text.ngram_size == 5, "Got " << text.ngram_size << " expected 5. Line number: " << i);
            //Decode word
            std::map<unsigned int, std::string>::iterator it;
            std::string word = pesho.decode_map.find(text.ngrams[1])->second;
            BOOST_CHECK_MESSAGE(word == "millennium", "Got " << word << " expected millennium. Line number: " << i);
        }

    }
}

BOOST_AUTO_TEST_CASE(trie_test) {
    std::pair<bool, std::string> res = test_trie(ARPA_TESTFILEPATH, 256);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Byte_array)

BOOST_AUTO_TEST_CASE(entry_byte_array_conversion_test) {
    Entry entry = {23, nullptr, 0.5, 0.75};
    Entry entry2 = {24, nullptr, 0.75, 0.5};
    std::vector<Entry> entryvec;
    entryvec.push_back(entry);
    entryvec.push_back(entry2);

    std::vector<unsigned char> byte_arr;
    byte_arr.reserve(2*getEntrySize());
    EntriesToByteArray(byte_arr, entryvec);

    Entry * new_entries = byteArrayToEntries(byte_arr, 2, 0);
    BOOST_CHECK_MESSAGE(entry.value == new_entries[0].value, "Got " << new_entries[0].value << " expected " << entry.value << ".");
    BOOST_CHECK_MESSAGE(entry.next_level == new_entries[0].next_level, "Got " << new_entries[0].next_level << " expected " << entry.next_level << ".");
    BOOST_CHECK_MESSAGE(entry.prob == new_entries[0].prob, "Got " << new_entries[0].prob << " expected " << entry.prob << ".");
    BOOST_CHECK_MESSAGE(entry.backoff == new_entries[0].backoff, "Got " << new_entries[0].backoff << " expected " << entry.backoff << ".");

    BOOST_CHECK_MESSAGE(entry2.value == new_entries[1].value, "Got " << new_entries[1].value << " expected " << entry2.value << ".");
    BOOST_CHECK_MESSAGE(entry2.next_level == new_entries[1].next_level, "Got " << new_entries[1].next_level << " expected " << entry2.next_level << ".");
    BOOST_CHECK_MESSAGE(entry2.prob == new_entries[1].prob, "Got " << new_entries[1].prob << " expected " << entry2.prob << ".");
    BOOST_CHECK_MESSAGE(entry2.backoff == new_entries[1].backoff, "Got " << new_entries[1].backoff << " expected " << entry2.backoff << ".");
    delete[] new_entries;
}

BOOST_AUTO_TEST_CASE(entry_byte_array_conversion_test_index) {
    bool pointer2Index = true;
    B_tree * boguspoiter = (B_tree *)123;
    Entry entry = {23, boguspoiter, 0.5, 0.75, 123};
    Entry entry2 = {24, boguspoiter, 0.6, 0.85, 124};
    std::vector<Entry> entryvec;
    entryvec.push_back(entry);
    entryvec.push_back(entry2);

    std::vector<unsigned char> byte_arr;
    byte_arr.reserve(2*getEntrySize(pointer2Index));
    EntriesToByteArray(byte_arr, entryvec, pointer2Index);

    Entry * new_entries = byteArrayToEntries(byte_arr, 2, 0, pointer2Index);
    BOOST_CHECK_MESSAGE(entry.value == new_entries[0].value, "Got " << new_entries[0].value << " expected " << entry.value << ".");
    BOOST_CHECK_MESSAGE(new_entries[0].next_level == nullptr, "Got " << new_entries[0].next_level << " expected " << (size_t)nullptr << ".");
    BOOST_CHECK_MESSAGE(entry.prob == new_entries[0].prob, "Got " << new_entries[0].prob << " expected " << entry.prob << ".");
    BOOST_CHECK_MESSAGE(entry.backoff == new_entries[0].backoff, "Got " << new_entries[0].backoff << " expected " << entry.backoff << ".");
    BOOST_CHECK_MESSAGE(entry.offset == new_entries[0].offset, "Got " << new_entries[0].offset << " expected " << entry.offset << ".");

    BOOST_CHECK_MESSAGE(entry2.value == new_entries[1].value, "Got " << new_entries[1].value << " expected " << entry.value << ".");
    BOOST_CHECK_MESSAGE(new_entries[1].next_level == nullptr, "Got " << new_entries[1].next_level << " expected " << (size_t)nullptr << ".");
    BOOST_CHECK_MESSAGE(entry2.prob == new_entries[1].prob, "Got " << new_entries[1].prob << " expected " << entry.prob << ".");
    BOOST_CHECK_MESSAGE(entry2.backoff == new_entries[1].backoff, "Got " << new_entries[1].backoff << " expected " << entry.backoff << ".");
    BOOST_CHECK_MESSAGE(entry2.offset == new_entries[1].offset, "Got " << new_entries[1].offset << " expected " << entry.offset << ".");
    delete[] new_entries;
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Trie_array)

BOOST_AUTO_TEST_CASE(Btree_trie_array_construct_test) {
    unsigned short btree_node_size = 256;
    std::pair<bool, std::string> res = test_byte_array_trie(ARPA_TESTFILEPATH, btree_node_size);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Serialization)

BOOST_AUTO_TEST_CASE(serialization_test) {
    std::string filepath = "/tmp/";
    const long double sysTime = time(0);
    std::stringstream s;
    s << filepath << sysTime; //Use random tmp directory

    //Create the btree_trie_array;
    LM out_lm;
    createTrieArray(ARPA_TESTFILEPATH, 256, out_lm);
    writeBinary(s.str(), out_lm);

    LM in_lm;
    readBinary(s.str(), in_lm);

    BOOST_CHECK_MESSAGE(in_lm.metadata == out_lm.metadata, "Mismatch in the read in and written metadata.");
    if (!(in_lm.metadata == out_lm.metadata)) { //BOOST_CHECK_MESSAGE doesn't work with overloaded ostreams so print separately.
        std::cout << "Read metadata:" << std::endl << in_lm.metadata << "Written metadata:" << std::endl << out_lm.metadata;
    }

    BOOST_CHECK_MESSAGE(out_lm.trieByteArray == in_lm.trieByteArray, "Read and written binary byte arrays differ.");
    BOOST_CHECK_MESSAGE(out_lm.encode_map == in_lm.encode_map, "Read and written encode maps differ.");
    BOOST_CHECK_MESSAGE(out_lm.decode_map == in_lm.decode_map, "Read and written decode maps differ.");

    boost::filesystem::remove_all(s.str());
}

BOOST_AUTO_TEST_SUITE_END()
