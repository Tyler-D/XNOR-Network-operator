add_executable(test_binarization ./test_binarization.cpp)
target_link_libraries(test_binarization cblas glog)
add_executable(test_xnorconv ./test_xnorconv.cpp)
target_link_libraries(test_xnorconv cblas glog)
