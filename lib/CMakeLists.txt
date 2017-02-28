#Produce python linkable library:
set(Python_ADDITIONAL_VERSIONS ${PYTHON_VER_FLAG})
find_package(PythonLibs)
if (${PYTHONLIBS_FOUND} AND DEFINED PYTHON_INCLUDE_DIR)
    find_package(Boost COMPONENTS serialization filesystem system python REQUIRED)
    add_library(ngrams_nematus SHARED python_bridge.cpp)
    if (DEFINED PYTHON_INCLUDE_DIR)
        target_include_directories(ngrams_nematus PUBLIC ${PYTHON_INCLUDE_DIR})
    endif()
    target_link_libraries(ngrams_nematus 
                        ${PYTHON_LIBRARIES} 
                        ${Boost_LIBRARIES}
                        memory_management_FPIC
                        gpu_search_v2_FPIC)
else()
    message("Python libraries not found, not building the shared python module.")
endif()

include(FindYamlCpp.cmake)
find_package(yaml-cpp)
if (${YAMLCPP_FOUND})
    message("yaml-cpp found, building fakeRNN")
    find_package(Boost COMPONENTS serialization filesystem system python REQUIRED)
    add_library(fakeRNN SHARED fakeRNN.cpp)
    target_link_libraries(fakeRNN 
                        ${YAMLCPP_LIBRARY} 
                        ${Boost_LIBRARIES}
                        memory_management_FPIC
                        gpu_search_v2_FPIC)
else()
    message("yaml-cpp not found on the system, not building fakeRNN")
endif()
