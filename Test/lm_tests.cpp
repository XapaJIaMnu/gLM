#include "tests_common.hh"
#include "trie_v2_impl.hh"
#include "lm_impl.hh"

BOOST_AUTO_TEST_SUITE(Trie_array)

BOOST_AUTO_TEST_CASE(Btree_trie_array_255) {
    unsigned short btree_node_size = 255;
    std::pair<bool, std::string> res = test_trie(ARPA_TESTFILEPATH, btree_node_size);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_trie_array_127) {
    unsigned short btree_node_size = 127;
    std::pair<bool, std::string> res = test_trie(ARPA_TESTFILEPATH, btree_node_size);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_trie_array_31) {
    unsigned short btree_node_size = 31;
    std::pair<bool, std::string> res = test_trie(ARPA_TESTFILEPATH, btree_node_size);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_trie_array_23) {
    unsigned short btree_node_size = 23;
    std::pair<bool, std::string> res = test_trie(ARPA_TESTFILEPATH, btree_node_size);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(LM_serialization)

BOOST_AUTO_TEST_CASE(LM_serialization_test) {
    std::string filepath = "/tmp/";
    const long double sysTime = time(0);
    std::stringstream s;
    s << filepath << sysTime; //Use random tmp directory

    //Create the btree_trie_array;
    unsigned short btree_node_size = 33;
    LM out_lm;
    createTrie(ARPA_TESTFILEPATH, out_lm, btree_node_size);
    out_lm.writeBinary(s.str());

    LM in_lm(s.str());

    BOOST_CHECK_MESSAGE(in_lm.metadata == out_lm.metadata, "Mismatch in the read in and written metadata.");
    if (!(in_lm.metadata == out_lm.metadata)) { //BOOST_CHECK_MESSAGE doesn't work with overloaded ostreams so print separately.
        std::cout << "Read metadata:" << std::endl << in_lm.metadata << "Written metadata:" << std::endl << out_lm.metadata;
    }

    BOOST_CHECK_MESSAGE(out_lm.trieByteArray == in_lm.trieByteArray, "Read and written binary btree trie arrays differ.");
    BOOST_CHECK_MESSAGE(out_lm.first_lvl == in_lm.first_lvl, "Read and written binary first_lvl arrays differ.");
    BOOST_CHECK_MESSAGE(out_lm.encode_map == in_lm.encode_map, "Read and written encode maps differ.");
    BOOST_CHECK_MESSAGE(out_lm.decode_map == in_lm.decode_map, "Read and written decode maps differ.");

    boost::filesystem::remove_all(s.str());
}

BOOST_AUTO_TEST_SUITE_END()
