#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, int* zero_element) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
    if(out[index] == 0) zero_element[index/CAFFE_CUDA_NUM_THREADS] += 1;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  //[houxiang]
  std::string filename = ("/home/hj14/caffe/hj_test/relu_sparsity.txt");
  std::ofstream sparsity_output;
  sparsity_output.open(filename.c_str(), ios::app);
  //count the zero number in each block to save space
  sparsity_output << count << " ";
  int block_num = CAFFE_GET_BLOCKS(count);
  int zero_cell[block_num];
  for(int i=0; i<block_num; ++i){
	  zero_cell[i] = 0;
  }
  cudaError_t err = cudaSuccess;
  int *dev_zero_cell;
  err = cudaMalloc((void**)&dev_zero_cell, block_num * sizeof(int));
  if(err!=cudaSuccess) {
        printf("the cudaMalloc on GPU is failed");
   }
  cudaMemcpy(dev_zero_cell, zero_cell, block_num * sizeof(int), cudaMemcpyHostToDevice);

  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope, dev_zero_cell );
  
  //[houxiang]
  cudaMemcpy(&zero_cell, dev_zero_cell, block_num * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_zero_cell);
  int total_zero = 0;
  for(int i=0; i<block_num; ++i){
	      total_zero = zero_cell[i] + total_zero;
        //sparsity_output << "[" <<i<<"]:"<< zero_cell[i]<<" ";
  }
  sparsity_output << total_zero << std::endl;

  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
