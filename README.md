# XNOR Network operators
raw implementation of XNOR convolution's operators including weights and input's binarization, im2col and binary convolution.

The idea is from [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279).
This is a version without using STL module, such as bitset and vector. Instead, the uint64_t arrays store binary inputs and weights.
Before the detailed optimization, the code already runs faster than atlas/openblas gemm (all in one single thread) and hardly ever loses the 
accuracy in some simple networks.
Still Working on openmp version. 

(tried xnor and xor, it seems no difference in both speed(single thread) and accuracy.I guess it is because the relu after the convolution)  

## speed test.

Enviroment |
----------|
Ubuntu 14.04| 
Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz|

Test: input 3x128x128; kernel 96x3x3; stride 1; padding 0

platform|speed/ms
------|-------
caffe-atlas| 19.3 |
caffe-openblas(OMP_THREAD_NUM=10) | 3.4 
caffe-xnor|5.5| 

## accuracy test.
[caffe example](https://github.com/Tyler-D/caffe-rc5_ex/tree/master/examples/xor)
