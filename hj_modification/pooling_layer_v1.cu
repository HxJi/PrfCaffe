#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

//add the attribute recording zero element number in every function interface to pass it out
//the last parameter for global functions

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask,int* zero_element) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    //[houxiang]
    int block_index = index/CAFFE_CUDA_NUM_THREADS;
    if(top_data[index] == 0) zero_element[block_index] += 1;
    //printf("%.6f,%i,%i\n%",top_data[index],blockIdx.x,zero_element[blockIdx.x]);

    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* zero_element) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
    //[houxiang]
    if(top_data[index] == 0) zero_element[blockIdx.x] += 1;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data, int* zero_element) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          //[houxiang]
          if(top_data[index] == 0) zero_element[blockIdx.x] += 1; 
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data, int* zero_element) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = 0.;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.;
    //[houxiang]
    if(top_data[index] == 0) zero_element[blockIdx.x] += 1;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  //[houxiang] sparsity output file name
  std::string filename = ("/home/hj14/caffe/hj_test/pooling_sparsity.txt");
  std::ofstream sparsity_output;
  sparsity_output.open(filename.c_str(), ios::app);
  int block_num = CAFFE_GET_BLOCKS(count);
  sparsity_output << count << " ";
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

  //count the zero number in each block to save space
  //sparsity_output <<"shape:" << train_data- << std::endl;
  //sparsity_output << "block_num:" << block_num << std::endl;
  //sparsity_output << "threads:" << CAFFE_CUDA_NUM_THREADS << std::endl;

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    printf("max_pool\n");
    //sparsity_output <<<< std::endl;
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask, dev_zero_cell);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    //sparsity_output << "ave_pool" << std::endl;
    printf("ave_pool\n");
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data, dev_zero_cell);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      //sparsity_output << "train_sto_pool" << std::endl;
      // We need to create the random index as well.
      printf("train_sto_pool\n");
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data,dev_zero_cell);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      //sparsity_output << "test_sto_pool" << std::endl;
      printf("test_sto_pool\n");
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data, dev_zero_cell);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }

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
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
