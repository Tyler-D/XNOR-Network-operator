#include"../xnet_function.h"
#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<sys/time.h>
#include<stack>
using namespace xnet;
using namespace std;
typedef struct data
{
  data(int n , int c, int h, int w, bool init): n_(n), c_(c),
  h_(h), w_(w){
    p_data_ = (float*)malloc(sizeof(float)*n*c*h*w);
    if(init){
    for(int i = 0 ; i < n ; i++)
      for(int j = 0; j < c ; j++)
        for(int k = 0; k < h ; k++)
          for(int l = 0; l < w ; l++)
            p_data_[i*c*h*w+j*h*w+k*w+l] = (i*c*h*w+j*h*w+k*w+l+1)*pow(-1, i*c*h*w+j*h*w+k*w+l); 
    }
    //for(int i = 0; i<n ; i++){
      //for(int j = 0; j < c*h*w; j++)
        //cout<<p_data_[i*c*h*w+j]<<" ";
      //cout<<endl;
    //}
  } 

  int n_;
  int c_;
  int h_;
  int w_;
  float* p_data_;
}Blob;
int main()
{
  //shape 
  int n = 96;
  int c = 3;
  int h = 300;
  int w = 300;
  int k = 3;
  int stride = 1;
  int pad = 0;
  int dilation = 1;

  Blob weights(n,c,k,k, true);
  int idx = 0; 
  //ifstream weight_file("/home/caffemaker/final_project/caffe_test/caffe_debug/rweights.out");
  //for(int row_idx = 0; row_idx < n; row_idx++)
    //for(int col_idx = 0; col_idx < c*k*k; col_idx++){
      //weight_file>>weights.p_data_[idx++];
  //}
  BinBlob<float> bin_weights(n,c,k,k);
  vector<float> alphas;
  alphas.resize(n);
  binarizeWeights(weights.p_data_, bin_weights,alphas);
  for(int i = 0 ; i< alphas.size(); i++)
    cout<<alphas[i];
  cout<<endl;
  const BinaryCode* weight_data = bin_weights.b_data();
  //uint64_t count = 0;
  //for(int i = 0; i<n; i++){
    //for(int j = 0; j<ceil((float)c*k*k/BIN_SIZE); j++){
        //uint64_t temp = weight_data[count++];
        //int cnt = 0;
        //while(cnt != BIN_SIZE){
          //cout<<(temp&1);
          //temp = temp>>1;
          //cnt++;
        //}
    //}
    //cout<<endl;
  //}
  
  //cout<<endl;
  //cout<<endl;

  Blob input(1,c,h,w, true);
  BinBlob<float> bin_input;
  //ifstream infile("/home/caffemaker/final_project/caffe_test/caffe_debug/rinput.out");
  //idx = 0; 
  //for(int row_idx = 0; row_idx < c; row_idx++)
    //for(int col_idx = 0; col_idx < h*w; col_idx++){
      //infile>>input.p_data_[idx++];
    //}
  //const BinaryCode* col_data = bin_input.b_data();
  //const int* shape = bin_input.shape();
  //for(int i = 0; i<shape[3]; i++)
    //for(int j = 0; j<ceil((float)shape[2]/BIN_SIZE); j++){
      //uint64_t temp = col_data[j*shape[3]+i];
      //int cnt = 0;
      //while(cnt != BIN_SIZE){
        //cout<<(temp&1);
        //temp = temp>>1;
        //cnt ++;
      //}
      //cout<<endl;
  //}


  int output_h = (h + 2*pad - (dilation*(k-1)+1))/stride + 1;
  int output_w = (w + 2*pad - (dilation*(k-1)+1))/stride + 1;
  
  float* output = (float*)malloc(sizeof(float)*n*output_h*output_w*n);

  struct timeval start,end;
  gettimeofday(&start, NULL);
  binarizeIm2Col(input.p_data_, bin_input, c, h, w, k, k, pad, pad, stride, stride,
     dilation, dilation);
  xorGEMM_baseline(n, ceil((float)c*k*k/BIN_SIZE), output_h*output_w,
                    bin_weights.b_data(), ceil((float)c*k*k/BIN_SIZE),
                    bin_input.b_data(), output_h*output_w,
                    output, output_h*output_w,
                    c*k*k, alphas);
  gettimeofday(&end, NULL);
  unsigned long diff = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000 ;
  cout<<"xor time: "<< diff<<"/ms"<<endl;
  
  gettimeofday(&start, NULL);
  binarizeIm2Col(input.p_data_, bin_input, c, h, w, k, k, pad, pad, stride, stride,
     dilation, dilation);
  xorGEMM_omp(n, ceil((float)c*k*k/BIN_SIZE), output_h*output_w,
                    bin_weights.b_data(), ceil((float)c*k*k/BIN_SIZE),
                    bin_input.b_data(), output_h*output_w,
                    output, output_h*output_w,
                    c*k*k, alphas);
  gettimeofday(&end, NULL);
  diff = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000 ;
  cout<<"xor time omp: "<< diff<<"/ms"<<endl;
  //ofstream bin_weight_out("/home/caffemaker/final_project/caffe_test/caffe_debug/test_weights.out");
  //const BinaryCode*  bin_weight = bin_weights.b_data(); 
  //idx = 0;
  //for(int row_idx = 0; row_idx < n; row_idx++){
    //for(int col_idx = 0; col_idx < ceil((float)(c*k*k)/BIN_SIZE); col_idx++){
      //uint64_t num = bin_weight[idx++];
      ////int cnt = 0;
      ////stack<int> st;
      ////while(cnt != BIN_SIZE){
        ////st.push(num&1);
        ////num = num>>1;
        ////cnt ++;
      ////}
      ////for(int i = 0; i < BIN_SIZE; i++){
        ////bin_weight_out<<st.top();
        ////st.pop();
      ////}
      //bin_weight_out<<num<<" ";
    //}
    //bin_weight_out<<endl;
  //}

  //ofstream bin_im_out("/home/caffemaker/final_project/caffe_test/caffe_debug/test_im.out");
  //const BinaryCode*  bin_im = bin_input.b_data(); 
  //idx = 0;
  //for(int row_idx = 0; row_idx < ceil((float)(c*k*k)/BIN_SIZE); row_idx++){
    //for(int col_idx = 0; col_idx < output_h*output_w ; col_idx++){
      //uint64_t num = bin_im[idx++];
      ////int cnt = 0;
      ////stack<int> st;
      ////while(cnt != BIN_SIZE){
        ////st.push(num&1);
        ////num = num>>1;
        ////cnt ++;
      ////}
      ////for(int i = 0; i < BIN_SIZE; i++){
        ////bin_weight_out<<st.top();
        ////st.pop();
      ////}
      //bin_im_out<<num<<" ";
    //}
    //bin_im_out<<endl;
  //}

  //ofstream outfile("/home/caffemaker/final_project/caffe_test/test.out");
  //idx = 0;
  //for(int row_idx = 0; row_idx < n ; row_idx++){
    //for(int col_idx = 0; col_idx < output_h*output_w; col_idx++){
      //outfile<<output[idx++]<<" "; 
    //}
    //outfile<<endl;
  //}
  //for(int i = 0; i<output_h; i++){
    //for(int j = 0; j< output_w; j++)
      //cout<<output[i*output_w+j]<<" ";
    //cout<<endl;
  //}

}
