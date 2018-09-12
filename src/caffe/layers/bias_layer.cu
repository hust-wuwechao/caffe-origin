#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BiasForward(const int n, const Dtype* in,
    const Dtype* bias, const int bias_dim, const int inner_dim,
    Dtype* out) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    //  index / inner_dim  得到对应的第几个通道 总的所用样本
    //  % bias_dim  得到真正的通道号码。
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  //  N*C*H*W
  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  // LOG(INFO)<<"bias_data"<<bias_data->shape_string();
  Dtype* top_data = top[0]->mutable_gpu_data();
  //  这里面进行发现他的数据是不是最优化的。
  BiasForward<Dtype>   // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (
      count, bottom_data, bias_data, bias_dim_, inner_dim_, top_data);
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {
  // propagate_down[0]=0  数据不需要反向 
  //LOG(INFO)<<"进入blas  反向 propagate_down[0]"<<propagate_down[0];
  //  2这应该相等的，按照
  //LOG(INFO)<<"bottom[0] != top[0]"<<bottom[0] != top[0];
  
  if (propagate_down[0] && bottom[0] != top[0]) 
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    //LOG(INFO)<<"top_diff"<<top_diff->shape_string();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff.
  //  明白是的。
  const bool bias_param = (bottom.size() == 1);
  //propagate_down[1]=0  
  //  this->param_propagate_down_[0】=1
  //LOG(INFO)<<"propagate_down[1]"<<propagate_down[1];
  //LOG(INFO)<<"this->param_propagate_down_[0]"<<this->param_propagate_down_[0];
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) 
   {
    //LOG(INFO)<<"bias_param"<<bias_param;
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_gpu_diff();
    bool accum = bias_param;
    //  32 个样本
    //  blas d的维度和通道的数目是一样的
    //LOG(INFO)<<"bias_dim_"<<bias_dim_;
    //  dim  ==N*H* W=100352
    // LOG(INFO)<<"dim_"<<dim_;
    for (int n = 0; n < outer_dim_; ++n) 
    {
      //  采用了 c*H*W  与  H*W  进乘法 bias_multiplier_=全部为1 
      caffe_gpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
          top_diff, bias_multiplier_.gpu_data(), Dtype(accum), bias_diff);
      top_diff += dim_;
      accum = true;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BiasLayer);

}  // namespace caffe
