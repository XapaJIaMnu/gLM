//#define ARPA_TESTFILEPATH is defined by cmake
#include "tests_common.hh"
#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"
#include <memory>
#include <boost/tokenizer.hpp>

 std::unique_ptr<float[]> sent2ResultsVector(std::string& sentence, LM& lm, unsigned char * btree_trie_gpu, unsigned int * first_lvl_gpu) {
    //tokenized
    boost::char_separator<char> sep(" ");
    std::vector<std::string> tokenized_sentence;
    boost::tokenizer<boost::char_separator<char> > tokens(sentence, sep);
    for (auto word : tokens) {
        tokenized_sentence.push_back(word);
    }

    //Convert to vocab IDs
    std::vector<unsigned int> vocabIDs = sent2vocabIDs(lm, tokenized_sentence, true);

    //Convert to ngram_queries to be parsed to the GPU
    std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, lm.metadata.max_ngram_order);

    //Now query everything on the GPU
    unsigned int num_keys = queries.size()/lm.metadata.max_ngram_order; //Only way to get how
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_keys, &results);

    searchWrapper(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    std::unique_ptr<float[]> results_cpu(new float[num_keys]);
    copyToHostMemory(results, results_cpu.get(), num_keys);

    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    return results_cpu;
}

 std::unique_ptr<float[]> sent2ResultsVector(std::string& sentence, GPUSearcher& engine, int streamID) {
    //tokenized
    boost::char_separator<char> sep(" ");
    std::vector<std::string> tokenized_sentence;
    boost::tokenizer<boost::char_separator<char> > tokens(sentence, sep);
    for (auto word : tokens) {
        tokenized_sentence.push_back(word);
    }

    //Convert to vocab IDs
    std::vector<unsigned int> vocabIDs = sent2vocabIDs(engine.lm, tokenized_sentence, true);

    //Convert to ngram_queries to be parsed to the GPU
    std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, engine.lm.metadata.max_ngram_order);

    //Now query everything on the GPU
    unsigned int num_keys = queries.size()/engine.lm.metadata.max_ngram_order; //Only way to get how
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_keys, &results);

    engine.search(gpuKeys, num_keys, results, streamID);

    //Copy back to host
    std::unique_ptr<float[]> results_cpu(new float[num_keys]);
    copyToHostMemory(results, results_cpu.get(), num_keys);

    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    return results_cpu;
}

std::pair<bool, unsigned int> checkIfSame(float * expected, float * actual, unsigned int num_entries) {
    bool all_correct = true;
    unsigned int wrong_idx = 0; //Get the index of the first erroneous element
    for (unsigned int i = 0; i < num_entries; i++) {
        if (!float_compare(expected[i], actual[i])) {
            wrong_idx = i;
            all_correct = false;
            break;
        }
    }

    return std::pair<bool, unsigned int>(all_correct, wrong_idx);
}

BOOST_AUTO_TEST_SUITE(Btree)

BOOST_AUTO_TEST_CASE(micro_LM_test_small)  {
    LM lm;
    createTrie(ARPA_TESTFILEPATH, lm, 7); //Use a small amount of entries per node.
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //Test whether we can find every single ngram that we stored
    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, ARPA_TESTFILEPATH);    
    BOOST_CHECK_MESSAGE(res.first, res.second);

    //Test if we have full queries and backoff working correctly with our toy dataset
    //The values that we have are tested against KenLM and we definitely get the same
    std::string sentence1 = "how are you doing today my really good man"; //Sentence with no backoff
    std::string sentence2 = "one oov word";
    std::string sentence3 = "a long sentence with many oov words and various other nicenesses";
    std::string sentence4 = "oov at beginning and oov at the end : unk";

    float expected1[10] = {-3.78869, -2.61558, -2.49612, -3.73654, -2.98147, -2.74779, -3.57897, -3.62742, -3.62742, -2.76108};
    float expected2[4] = {-2.92395, -3.85537, -3.54566, -2.76108};
    float expected3[12] = {-2.50186, -3.54072, -3.76045, -2.14939, -3.3258, -3.76045, -3.67869, -1.68129, -3.77999, -3.15575, -3.7833, -2.67932};
    float expected4[11] = {-4.27026, -2.47602, -3.93291, -1.68129, -3.91301, -2.47602, -0.500325, -3.22683, -3.31373, -3.76045, -2.67932};

    //Query on the GPU
    std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, lm, btree_trie_gpu, first_lvl_gpu);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 10);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);

    is_correct = checkIfSame(expected2, res_2.get(), 4);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 12);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 11);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

    //Free GPU memory now:
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);

    //@TODO Tho dataset for the test is likely too small to expose any race conditions in our code. We need a test with a big dataset
    //that can really do that. After IO is fixed and presumably binarization doesn't take forever we should do that.

}

BOOST_AUTO_TEST_CASE(micro_LM_test_small_serialization)  {
    LM lm_orig;
    createTrie(ARPA_TESTFILEPATH, lm_orig, 7); //Use a small amount of entries per node.

    std::string filepath = "/tmp/";
    const long double sysTime = time(0);
    std::stringstream s;
    s << filepath << sysTime; //Use random tmp directory

    lm_orig.writeBinary(s.str());
    LM lm(s.str());

    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //Test whether we can find every single ngram that we stored
    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, ARPA_TESTFILEPATH);    
    BOOST_CHECK_MESSAGE(res.first, res.second);

    //Test if we have full queries and backoff working correctly with our toy dataset
    //The values that we have are tested against KenLM and we definitely get the same
    std::string sentence1 = "how are you doing today my really good man"; //Sentence with no backoff
    std::string sentence2 = "one oov word";
    std::string sentence3 = "a long sentence with many oov words and various other nicenesses";
    std::string sentence4 = "oov at beginning and oov at the end : unk";

    float expected1[10] = {-3.78869, -2.61558, -2.49612, -3.73654, -2.98147, -2.74779, -3.57897, -3.62742, -3.62742, -2.76108};
    float expected2[4] = {-2.92395, -3.85537, -3.54566, -2.76108};
    float expected3[12] = {-2.50186, -3.54072, -3.76045, -2.14939, -3.3258, -3.76045, -3.67869, -1.68129, -3.77999, -3.15575, -3.7833, -2.67932};
    float expected4[11] = {-4.27026, -2.47602, -3.93291, -1.68129, -3.91301, -2.47602, -0.500325, -3.22683, -3.31373, -3.76045, -2.67932};

    //Query on the GPU
    std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, lm, btree_trie_gpu, first_lvl_gpu);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 10);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);

    is_correct = checkIfSame(expected2, res_2.get(), 4);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 12);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 11);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

    //Free GPU memory now:
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);

    boost::filesystem::remove_all(s.str());
    //@TODO Tho dataset for the test is likely too small to expose any race conditions in our code. We need a test with a big dataset
    //that can really do that. After IO is fixed and presumably binarization doesn't take forever we should do that.

}


BOOST_AUTO_TEST_CASE(micro_LM_test_large)  {
    LM lm;
    createTrie(ARPA_TESTFILEPATH, lm, 127); //Use a large amount of entries per node
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    //Test whether we can find every single ngram that we stored
    std::pair<bool, std::string> res = testQueryNgrams(lm, btree_trie_gpu, first_lvl_gpu, ARPA_TESTFILEPATH);    
    BOOST_CHECK_MESSAGE(res.first, res.second);

    //Test if we have full queries and backoff working correctly with our toy dataset
    //The values that we have are tested against KenLM and we definitely get the same
    std::string sentence1 = "how are you doing today my really good man"; //Sentence with no backoff
    std::string sentence2 = "one oov word";
    std::string sentence3 = "a long sentence with many oov words and various other nicenesses";
    std::string sentence4 = "oov at beginning and oov at the end : unk";

    float expected1[10] = {-3.78869, -2.61558, -2.49612, -3.73654, -2.98147, -2.74779, -3.57897, -3.62742, -3.62742, -2.76108};
    float expected2[4] = {-2.92395, -3.85537, -3.54566, -2.76108};
    float expected3[12] = {-2.50186, -3.54072, -3.76045, -2.14939, -3.3258, -3.76045, -3.67869, -1.68129, -3.77999, -3.15575, -3.7833, -2.67932};
    float expected4[11] = {-4.27026, -2.47602, -3.93291, -1.68129, -3.91301, -2.47602, -0.500325, -3.22683, -3.31373, -3.76045, -2.67932};

    //Query on the GPU
    std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, lm, btree_trie_gpu, first_lvl_gpu);
    std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, lm, btree_trie_gpu, first_lvl_gpu);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 10);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);

    is_correct = checkIfSame(expected2, res_2.get(), 4);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 12);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 11);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

    //Free GPU memory now:
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);

    //@TODO Tho dataset for the test is likely too small to expose any race conditions in our code. We need a test with a big dataset
    //that can really do that. After IO is fixed and presumably binarization doesn't take forever we should do that.

}

BOOST_AUTO_TEST_CASE(micro_LM_test_normal_streams)  {
    LM lm;
    createTrie(ARPA_TESTFILEPATH, lm, 31); //Use a large amount of entries per node
    
    GPUSearcher engine(2, lm);

    //Test if we have full queries and backoff working correctly with our toy dataset
    //The values that we have are tested against KenLM and we definitely get the same
    std::string sentence1 = "how are you doing today my really good man"; //Sentence with no backoff
    std::string sentence2 = "one oov word";
    std::string sentence3 = "a long sentence with many oov words and various other nicenesses";
    std::string sentence4 = "oov at beginning and oov at the end : unk";

    float expected1[10] = {-3.78869, -2.61558, -2.49612, -3.73654, -2.98147, -2.74779, -3.57897, -3.62742, -3.62742, -2.76108};
    float expected2[4] = {-2.92395, -3.85537, -3.54566, -2.76108};
    float expected3[12] = {-2.50186, -3.54072, -3.76045, -2.14939, -3.3258, -3.76045, -3.67869, -1.68129, -3.77999, -3.15575, -3.7833, -2.67932};
    float expected4[11] = {-4.27026, -2.47602, -3.93291, -1.68129, -3.91301, -2.47602, -0.500325, -3.22683, -3.31373, -3.76045, -2.67932};

    //Query on the GPU
    std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, engine, 1);
    std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, engine, 0);
    std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, engine, 1);
    std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, engine, 0);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 10);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);

    is_correct = checkIfSame(expected2, res_2.get(), 4);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 12);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 11);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

}

BOOST_AUTO_TEST_CASE(micro_LM_test_exp_streams)  {
    LM lm;
    createTrie(ARPA_TESTFILEPATH, lm, 31); //Use a large amount of entries per node
    
    GPUSearcher engine(2, lm, true);

    //Test if we have full queries and backoff working correctly with our toy dataset
    //The values that we have are tested against KenLM and we definitely get the same
    std::string sentence1 = "how are you doing today my really good man"; //Sentence with no backoff
    std::string sentence2 = "one oov word";
    std::string sentence3 = "a long sentence with many oov words and various other nicenesses";
    std::string sentence4 = "oov at beginning and oov at the end : unk";

    float expected1[10] = {0.02262, 0.07312, 0.08240, 0.02383, 0.05071, 0.06406, 0.02790, 0.02658, 0.02658, 0.06322};
    float expected2[4] = {0.05372, 0.02116, 0.02884, 0.06322};
    float expected3[12] = {0.08193, 0.02899, 0.02327, 0.11655, 0.03594, 0.02327, 0.02525, 0.18613, 0.02282, 0.04260, 0.02274, 0.06860};
    float expected4[11] = {0.01397, 0.08407, 0.01958, 0.18613, 0.01998, 0.08407, 0.60633, 0.03968, 0.03638, 0.02327, 0.06860};

    //Query on the GPU
    std::unique_ptr<float[]> res_1 = sent2ResultsVector(sentence1, engine, 1);
    std::unique_ptr<float[]> res_2 = sent2ResultsVector(sentence2, engine, 0);
    std::unique_ptr<float[]> res_3 = sent2ResultsVector(sentence3, engine, 1);
    std::unique_ptr<float[]> res_4 = sent2ResultsVector(sentence4, engine, 0);

    //Check if the results are as expected
    std::pair<bool, unsigned int> is_correct = checkIfSame(expected1, res_1.get(), 10);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 1: Expected: "
        << expected1[is_correct.second] << ", got: " << res_1[is_correct.second]);

    is_correct = checkIfSame(expected2, res_2.get(), 4);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 2: Expected: "
        << expected2[is_correct.second] << ", got: " << res_2[is_correct.second]);

    is_correct = checkIfSame(expected3, res_3.get(), 12);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 3: Expected: "
        << expected3[is_correct.second] << ", got: " << res_3[is_correct.second]);

    is_correct = checkIfSame(expected4, res_4.get(), 11);
    BOOST_CHECK_MESSAGE(is_correct.first, "Error! Mismatch at index " << is_correct.second << " in sentence number 4: Expected: "
        << expected4[is_correct.second] << ", got: " << res_4[is_correct.second]);

}

BOOST_AUTO_TEST_SUITE_END()
