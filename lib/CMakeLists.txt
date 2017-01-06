#Produce python linkable library:
find_package(Boost COMPONENTS serialization filesystem system python REQUIRED)
set(Python_ADDITIONAL_VERSIONS ${PYTHON_VER_FLAG})
find_package(PythonLibs)
if (${PYTHONLIBS_FOUND})
    add_library(ngrams_nematus SHARED python_bridge.cpp)
    target_link_libraries(ngrams_nematus 
                        ${PYTHON_LIBRARIES} 
                        ${Boost_LIBRARIES}
                        memory_management_FPIC
                        gpu_search_v2_FPIC)
else()
    message("Python libraries not found, not building the shared python module.")
endif()
