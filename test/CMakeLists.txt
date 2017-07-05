add_executable(test_binarization ./test_binarization.cpp)
target_link_libraries(test_binarization cblas glog)
add_executable(test_xorconv ./test_xorconv.cpp)
target_link_libraries(test_xorconv cblas glog)
