#include "tests_common.hh"
#include "btree_v2_impl.hh"

BOOST_AUTO_TEST_SUITE(Btree)
BOOST_AUTO_TEST_CASE(Btree_small_lastngram) {
    unsigned int num_elements = 25;
    unsigned int BtreeNodeSize = 5;
    bool lastNgram = true;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_small_innerngram) {
    unsigned int num_elements = 25;
    unsigned int BtreeNodeSize = 5;
    bool lastNgram = false;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_large_lastngram) {
    unsigned int num_elements = 150321;
    unsigned int BtreeNodeSize = 33;
    bool lastNgram = true;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_large_innerngram) {
    unsigned int num_elements = 150321;
    unsigned int BtreeNodeSize = 33;
    bool lastNgram = false;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_trivial_lastngram) {
    unsigned int num_elements = 20;
    unsigned int BtreeNodeSize = 33;
    bool lastNgram = true;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_trivial_innerngram) {
    unsigned int num_elements = 20;
    unsigned int BtreeNodeSize = 33;
    bool lastNgram = false;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_semitrivial_lastngram) {
    unsigned int num_elements = 45;
    unsigned int BtreeNodeSize = 33;
    bool lastNgram = true;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_CASE(Btree_semitrivial_innerngram) {
    unsigned int num_elements = 45;
    unsigned int BtreeNodeSize = 33;
    bool lastNgram = false;
    std::pair<bool, std::string> res = test_btree_v2(num_elements, BtreeNodeSize, lastNgram);
    BOOST_CHECK_MESSAGE(res.first, res.second);
}

BOOST_AUTO_TEST_SUITE_END()
