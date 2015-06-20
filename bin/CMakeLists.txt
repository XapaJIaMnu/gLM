find_package(Boost COMPONENTS serialization filesystem system REQUIRED)

add_executable(binarize binarize.cpp)
add_executable(query query.cpp)

target_link_libraries(binarize
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
		     )

target_link_libraries(query
                      ${Boost_SERIALIZATION_LIBRARY}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
		     )
