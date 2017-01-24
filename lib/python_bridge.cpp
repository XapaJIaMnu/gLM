#include "nematus_ngrams.hh" 
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
using namespace boost::python;

boost::python::object testNDARRAY(long softmax_size, long sentence_length, long batch_size) {
    float * result = new float[softmax_size*sentence_length*batch_size];

    //Simulate population of a our sentences
    int idx = 0;
    for (int i = 0; i<batch_size; i++) {
    	for (int j = 0; j<sentence_length; j++) {
    		for (int n=0; n<softmax_size; n++) {
    			result[idx] = (float)(i + ((float)j)/10 + ((float)n)/100);
    			idx++;
    		}
    	}
    }

    npy_intp shape[3] = { softmax_size, sentence_length, batch_size }; // array size
    npy_intp strides[3] = {softmax_size*4, sentence_length*4, 4};
    /*PyObject* obj = PyArray_SimpleNewFromData(3, shape, NPY_FLOAT, result); */
    PyObject* obj = PyArray_New(&PyArray_Type, 3, shape, NPY_FLOAT, // data type
                              strides, result, // data pointer
                              0, NPY_ARRAY_CARRAY, // NPY_ARRAY_CARRAY_RO for readonly
                              NULL);
    handle<> array( obj );
    return object(array);
}


BOOST_PYTHON_MODULE(libngrams_nematus)
{
    //Without this you get a segfault inside numpy...
    import_array();
    class_<NematusLM>("NematusLM", init<char *, char *, unsigned int, int>())
        .def("processBatch", &NematusLM::processBatchNDARRAY)
        .def("getLastNumQueries", &NematusLM::getLastNumQueries)
        .def("freeResultsMemory", &NematusLM::freeResultsMemory)
        .def("testNDARRAY", &testNDARRAY)
        .staticmethod("testNDARRAY")
    ;
}
