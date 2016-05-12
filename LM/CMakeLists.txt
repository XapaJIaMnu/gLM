find_package(Boost COMPONENTS serialization filesystem system REQUIRED)

add_library(gLM STATIC gpuLM.cpp) 
                      
target_link_libraries(gLM
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      )
