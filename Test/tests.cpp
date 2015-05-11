#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Suites
//#define ARPA_TESTFILEPATH is defined by cmake
#define FLOAT_TOLERANCE 1e-5*1e-5
#include <boost/test/unit_test.hpp>
#include "btree.hh"
#include "tokenizer.hh"

//Float comparison
inline bool float_compare(float a, float b) { 

    return (a - b) * (a - b) < FLOAT_TOLERANCE;
}

//Init b_tree and the set of numbers that it contains
std::pair<B_tree *, std::set<unsigned int> > init_btree(int max_degree, int num_entries) {

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
    return std::pair<B_tree *, std::set<unsigned int> >(pesho, prev_nums);
}

BOOST_AUTO_TEST_SUITE(Entry_overload)
BOOST_AUTO_TEST_CASE(various_tests) {
    Entry A = {10, nullptr, false};
    Entry B = {15, nullptr, false};
    Entry C = {10, nullptr, false};
    Entry D = {16, nullptr, false};
    Entry E = {17, nullptr, false};
    Entry G = {19, nullptr, false};

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
    Entry elem ={*it, nullptr, false};
    res = pesho->find_element(elem);
    value = res.first->words[res.second].value;
    BOOST_CHECK_MESSAGE(*it == value, "Expected: " << *it << ", got: " << value << " at position: " << position << "." );

    //Second case
    it = prev_nums.begin();
    position += 15;
    std::advance(it, position);
    Entry elem2 = {*it, nullptr, false};
    res2 = pesho->find_element(elem2);
    value = res2.first->words[res2.second].value;
    BOOST_CHECK_MESSAGE(*it == value, "Expected: " << *it << ", got: " << value << " at position: " << position << "." );

    delete pesho;
}

BOOST_AUTO_TEST_CASE(very_big_btree) {
    //init
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(17, 65000);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    //test
    std::pair<bool, std::string> test_result = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result.first, test_result.second);

    //Test compression:
    pesho->compress();
    std::pair<bool, std::string> test_result_compressed = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result_compressed.first, "Compressed: " << test_result_compressed.second);
    delete pesho;
}

BOOST_AUTO_TEST_CASE(very_big_btree_small_node) {
    //init
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(5, 15000);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    //test
    std::pair<bool, std::string> test_result = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result.first, test_result.second);

    //Test compression:
    pesho->compress();
    std::pair<bool, std::string> test_result_compressed = test_btree(prev_nums, pesho);
    BOOST_CHECK_MESSAGE(test_result_compressed.first, "Compressed: " << test_result_compressed.second);
    delete pesho;
}

BOOST_AUTO_TEST_CASE(modify_entry_test) {
    //Get the b_tree and the set
    std::pair<B_tree *, std::set<unsigned int> > initialization = init_btree(3, 25);
    B_tree * pesho = initialization.first;
    std::set<unsigned int> prev_nums = initialization.second;

    //Test search
    int position = 5;
    bool last;
    std::set<unsigned int>::iterator it = prev_nums.begin();
    std::advance(it, position);
    std::pair<B_tree_node *, int> res;
    std::pair<B_tree_node *, int> res2;


    Entry elem ={*it, nullptr, false};
    res = pesho->find_element(elem);
    last = res.first->words[res.second].last;
    //Change the value now
    res.first->words[res.second].last = !last;

    //Find the node again and check if the change was recorded
    Entry elem2 = {*it, nullptr, false};
    res2 = pesho->find_element(elem2);
    bool current_last = res2.first->words[res2.second].last;
    BOOST_CHECK_MESSAGE(current_last == !last, "Expected: " << !last << ", got: " << last << " at position: " << position << "." );
    delete pesho;
}
 
BOOST_AUTO_TEST_SUITE_END()
 
BOOST_AUTO_TEST_SUITE(Parser)
 
BOOST_AUTO_TEST_CASE(max_ngrams_test) {
    ArpaReader pesho(ARPA_TESTFILEPATH);
    BOOST_CHECK_MESSAGE(pesho.max_ngrams == 4, "Wrong number of max ngrams. Got: " << pesho.max_ngrams << ", expected 4.");

    //Just iterate till the end of the file so that we don't have multiple file handles.
    processed_line text = pesho.readline();
    while (!text.filefinished){
        text = pesho.readline();
    }
}
 
BOOST_AUTO_TEST_CASE(random_arpa_lines_test) {
    ArpaReader pesho(ARPA_TESTFILEPATH);

    //Do several tests based
    for (int i = 0; i < 18; i++) {
        processed_line text = pesho.readline();
        if (i == 3){
            BOOST_CHECK_MESSAGE(float_compare(text.backoff, -0.2553), "Got " << text.backoff << " expected -0.2553.");
        }

        if (i == 6){
            BOOST_CHECK_MESSAGE(float_compare(text.score, -0.6990), "Got " << text.score << " expected -0.6990.");
        }

        if (i == 13){
            BOOST_CHECK_MESSAGE(text.ngram_size == 2, "Got " << text.ngram_size << " expected 2.");
            //Decode word
            std::map<unsigned int, std::string>::iterator it;
            std::string word = pesho.decode_map.find(text.ngrams[1])->second;
            BOOST_CHECK_MESSAGE(word == "wood", "Got " << word << " expected 2.");
        }

        if (i == 17){
            BOOST_CHECK_MESSAGE(float_compare(text.score, -0.1213), "Got " << text.score << " expected -0.1213.");
            //Decode word
            std::map<unsigned int, std::string>::iterator it;
            std::string word = pesho.decode_map.find(text.ngrams[3])->second;
            BOOST_CHECK_MESSAGE(word == "jean", "Got " << word << " expected 2.");
        }

    }
}

BOOST_AUTO_TEST_SUITE_END()
