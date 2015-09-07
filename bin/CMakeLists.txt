find_package(Boost COMPONENTS serialization filesystem iostreams system REQUIRED)
find_package(BZip2 REQUIRED)

add_executable(binarize binarize.cpp)
add_executable(batch_query batch_query.cpp)

target_link_libraries(binarize
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_IOSTREAMS_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${BZIP2_LIBRARIES}
                     )

target_link_libraries(batch_query
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_IOSTREAMS_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
                      ${BZIP2_LIBRARIES}
                      memory_management
                      gpu_search
                     )
