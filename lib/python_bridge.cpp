#include "nematus_ngrams.hh" 
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(libngrams_nematus)
{
    //Without this you get a segfault inside numpy...
    import_array();
    class_<NematusLM>("NematusLM", init<char *, char *, unsigned int, int>())
        .def("processBatch", &NematusLM::processBatchNDARRAY)
        .def("getLastNumQueries", &NematusLM::getLastNumQueries)
        .def("freeResultsMemory", &NematusLM::freeResultsMemory)
    ;
}
