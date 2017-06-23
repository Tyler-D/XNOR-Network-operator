#include"../xnet_function.h"
#include<iostream>
#include<stdlib.h>
#include<math.h>
using namespace xnet;
using namespace std;
int main()
{
  //shape 
  int n = 1;
  int c = 3;
  int h = 4;
  int w = 4;

  float* data = (float*)malloc(sizeof(float)*(n*c*h*w));
  for(int i = 0 ; i < n ; i++)
    for(int j = 0; j < c ; j++)
      for(int k = 0; k < h ; k++)
        for(int l = 0; l < w ; l++)
          data[i*c*h*w+j*h*w+k*w+l] = (i*c*h*w+j*h*w+k*w+l+1)*pow(-1, i*c*h*w+j*h*w+k*w+l); 

  for(int i = 0; i<n ; i++){
    for(int j = 0; j < c*h*w; j++)
      cout<<data[i*c*h*w+j]<<" ";
    cout<<endl;
  }
  cout<<endl;
  cout<<endl;
    
  BinBlob<float> weight_blob(n, c, h, w);

  vector<float> alpha;
  alpha.resize(n);
  binarizeWeights(&data[0], weight_blob, alpha);
  const BinaryCode* bin_data = weight_blob.b_data();
  uint64_t count = 0;
  for(int i = 0; i<n; i++){
    for(int j = 0; j<ceil((float)c*h*w/64); j++){
        uint64_t temp = bin_data[count++];
        int cnt = 0;
        while(cnt != 64){
          cout<<(temp&1);
          temp = temp>>1;
          cnt++;
        }
    }
    cout<<endl;
  }

  cout<<endl;
  cout<<endl;
  BinBlob<float> col_buf;
  binarizeIm2Col(data, col_buf, 3, 4, 4, 2, 2, 0, 0, 1, 1, 1, 1);
  const BinaryCode* col_data = col_buf.b_data();
  const int* shape = col_buf.shape();
  for(int i = 0; i<shape[3]; i++)
    for(int j = 0; j<ceil((float)shape[2]/BIN_SIZE); j++){
      uint64_t temp = col_data[j*shape[3]+i];
      int cnt = 0;
      while(cnt != BIN_SIZE){
        cout<<(temp&1);
        temp = temp>>1;
        cnt ++;
      }
      cout<<endl;
  }
  //const vector<BinBlock>& nbin_data = input_blob.bin_data();
  //for(int i = 0; i<nbin_data.size(); i++){
    //for(int j = 0; j < nbin_data[i].size(); j++)
      //cout<<nbin_data[i][j]<<" ";
    //cout<<endl;
  //}
}
