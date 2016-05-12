#pragma once
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Suites
#define FLOAT_TOLERANCE 1e-5*1e-5
#include <boost/test/unit_test.hpp> 

//Float comparison
inline bool float_compare(float a, float b) { 

    return (a - b) * (a - b) < FLOAT_TOLERANCE;
}
