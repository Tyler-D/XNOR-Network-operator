#ifndef BINARY_COMMON_HEADER
#define BINARY_COMMON_HEADER

#define SUCCESS 1
#define FAILED 0
//#include<vector>
//#include<bitset>
//#include<memory>
#include<stdlib.h>
#include<string.h>
#include<omp.h>
#include<math.h>
#include<glog/logging.h>


//using namespace std;

//@DEPRECATED(Raven) dynamic bitset map. Considering the kernel count and input count cannot
//perfectly fit 64 bit, we may have zero padding in fixed size bitset.
//And the speed of popcnt operations, which relates to the hardware architecture,
//doesn't has linear relation with the bit size. Therefore, maybe we can use
//dynamic bitset here.

//@DEPRECATED(Raven): Use uint8_t to store data 
//@TODO(Raven): use uint64_t array to store binary data
//@TODO(Raven): support group convolution.
#define BIN_SIZE 64

namespace xnet{


typedef uint64_t BinaryCode;

template <typename Dtype>
class BinBlob final{
  public:
    BinBlob()
      :count_(0), b_data_(NULL), binary_count_(0) {}

    ~BinBlob(){
      if (b_data_ != NULL)
        free(b_data_);
    } 

    explicit BinBlob(const int num, const int channels, const int height,
        const int width):b_data_(NULL), binary_count_(0){
        shape_[0] = num;
        shape_[1] = channels;
        shape_[2] = height;
        shape_[3] = width;

        //bin_data_.resize(num);
        
        count_ = num * channels * height * width;
    }

    void Reshape(const int num, const int channels, const int height,
        const int width){
      shape_[0] = num;
      shape_[1] = channels;
      shape_[2] = height;
      shape_[3] = width;

      count_ = num * channels * height * width; 
      //if (b_data_ != NULL)
        //free(b_data_);
    }
    
    //count: the number of uint64_t to store the bits.
    int allocateBC(uint64_t count, bool init){
      if (count == 0 )
        return SUCCESS;


      if (count > binary_count_){
        binary_count_ = count;
        if (b_data_ != NULL)
          free(b_data_);
        // allocate memory 
        b_data_ = (uint64_t*)malloc(sizeof(uint64_t)*count); 
      }
      
      // set 0
      // ALWAYS REMEMBER: @TODO FUCK THIS MEMSET, THE THIRD PARAMETER 
      // IS SIZE_T; 
      if (init){
        memset(b_data_, 0, sizeof(uint64_t)*count);
      }

      if (b_data_ == NULL) 
        return FAILED; 
      else 
        return SUCCESS;
    }

    const int* shape() {
      return shape_;
    }

    const BinaryCode* b_data(){
      return b_data_;
    }

    const uint64_t count(){
      return count_;
    }

    BinaryCode* mutable_b_data(){
      return b_data_;
    }

    BinBlob<Dtype>& operator=(const BinBlob<Dtype>&) = delete;


  protected:
       uint64_t  count_;
       uint64_t binary_count_;
       //real value shape NCHW
       int shape_[4];
       //binary data
       BinaryCode* b_data_; 
};

} 

#endif 
