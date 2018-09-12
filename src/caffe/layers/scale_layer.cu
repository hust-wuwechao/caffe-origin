#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScaleForward(const int n, const Dtype* in,
    const Dtype* scale, const int scale_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

template <typename Dtype>
__global__ void ScaleBiasForward(const int n, const Dtype* in,
    const Dtype* scale, const Dtype* bias,
    const int scale_dim, const int inner_dim, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    const int scale_index = (index / inner_dim) % scale_dim;
    // 这里面再次很明确的说明，
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
 //  LOG(INFO)<<"正向  bottom[0] == top[0]"<<bottom[0] == top[0];
  //  假设输入为  32*2048*7*7
  if (bottom[0] == top[0]) 
  {
    //  in-place computation; need to store bottom data before overwriting it.
    //  Note that this is only necessary for Backward; we could skip this if not
    //  doing Backward, but Caffe currently provides no way of knowing whether
    //  we'll need to do Backward at the time of the Forward call.
    //  拷贝到零时空间
    //  
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
               temp_.mutable_gpu_data());
  }
  // LOG(INFO)<<"bottom.size()"<<bottom.size();

  const Dtype* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // LOG(INFO)<<" bias_layer_ "<<bias_layer_==NULL;
  if (bias_layer_) 
  {
     const Dtype* bias_data = this->blobs_[bias_param_id_]->gpu_data();
     //  按照通道进行相称加上偏执
     ScaleBiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, bias_data, scale_dim_, inner_dim_,
        top_data);
  }
  else 
  {
    //   不加上---偏执
    ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_data, scale_dim_, inner_dim_, top_data);
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //   如果 blas  层存在
  //   参数需要重新新计算
  if (bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
  }
  // 假设输入为  32 2048 7 7 (3211264)
  //LOG(INFO)<<"inner_dim_"<<inner_dim_;    //    49
  //LOG(INFO)<<"outer_dim_"<<outer_dim_;    //    32
  //LOG(INFO)<<"bias_layer_"<<bias_layer_;  //    1
  //     1 
  //LOG(INFO)<<"this->param_propagate_down_[this->param_propagate_down_.size() - 1]"<<this->param_propagate_down_[this->param_propagate_down_.size() - 1];
  const bool scale_param = (bottom.size() == 1);
  //    1  具有 scale参数
  //    LOG(INFO)<<"scale_param"<<scale_param;
  Blob<Dtype>* scale = scale_param ? this->blobs_[0].get() : bottom[1];
  //    0
  //  LOG(INFO)<<"!scale_param && propagate_down[1]"<<!scale_param && propagate_down[1];
  //   0   
   // LOG(INFO)<<"propagate_down[1]"<<propagate_down[1];
  //   1
    //LOG(INFO)<<"scale_param "<<scale_param;
  //   1
    //LOG(INFO)<<"this->param_propagate_down_[0]"<<this->param_propagate_down_[0];
  //  满足1
  //  参数需要被反向传递？是的。
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) 
    {    
    const Dtype* top_diff = top[0]->gpu_diff();
    const bool in_place = (bottom[0] == top[0]);
    // 1 满足是原地操作
    //LOG(INFO)<<"in_place"<<in_place;
    //  从temp里面获取内容
    const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->gpu_data();
    // 
    //LOG(INFO)<<"bottom[0]->count()   "<<bottom[0]->count();
    // 通道的数目和总的样本个数自然不一样。
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    //  0
    //LOG(INFO)<<"is_eltwise  "<<is_eltwise;
    //  积德数值为  temp_.mutable_gpu_data()
    //  也即是原来的值
    Dtype* product = (is_eltwise ? scale->mutable_gpu_diff() :
        (in_place ? temp_.mutable_gpu_data() : bottom[0]->mutable_gpu_diff()));
    //  top[0]->count=
     // LOG(INFO)<<"top[0]->count()"<<top[0]->count();
    //   采用按照elem-wise 进行乘法
    //   top[0]->count() 3211264  ==N*C*H*W
    //   得到得到每一个
    caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) 
     {
      //  
      Dtype* sum_result = NULL;
      //LOG(INFO)<<"sum_result_.count()  "<<sum_result_.count();
      if (inner_dim_ == 1) 
      { 
        sum_result = product;
      } 
      else if (sum_result_.count() == 1)  //65536==N*C 也就是说sum-result等于多个样本多个通道的和值
      {
       
        const Dtype* sum_mult = sum_multiplier_.gpu_data();
        //  获得 scale值得梯度，和通道的数目应该是一样的，也就是，2048
        Dtype* scale_diff = scale->mutable_cpu_diff();
        //LOG(INFO)<<"scale_param"<<scale_param;
        if (scale_param) 
        {
          //  存在scale 参数
          Dtype result;
          //  7*7  进行
          caffe_gpu_dot(inner_dim_, product, sum_mult, &result);
          //  结果进行相加。
          *scale_diff += result;
        }
         else 
        {
          caffe_gpu_dot(inner_dim_, product, sum_mult, scale_diff);
        }
      } 
      else 
      {
        const Dtype* sum_mult = sum_multiplier_.gpu_data();
        //    outer_dim_ == 32 
        //    sum_result= sum_result_.mutable_gpu_data()
        //    N*C
        //    65536
        //LOG(INFO)<<"sum_result_.count()   AAAAA"<<sum_result_.count();
        //    49
        //LOG(INFO)<<"inner_dim_   AAAAA"<<inner_dim_;
        // sum_result_.mutable_gpu_data()
        sum_result = (outer_dim_ == 1) ?
            scale->mutable_gpu_diff() : sum_result_.mutable_gpu_data();
        //  product  等于原本的TOP【0】
        //  sum_mult=H*C *1 表示所有的通道的和值
        //  得到结果为  sum_result=N*C  
        //  记录每一个样本每一个通道的和值
        //  得到每一个通道的相对于 scale的梯度的累加。
        //  (N*C)*(H*W)    (H*W)*1    sum_mult 应该等于1的H*W的向量 吧
        caffe_gpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      if (outer_dim_ != 1) 
      {
        //LOG(INFO)<<"进入这里";
        const Dtype* sum_mult = sum_multiplier_.gpu_data();
        // 2048  表示通道的个数。每一个通道具有一个值
        //LOG(INFO)<<"scale_dim"<<scale_dim_;
        if (scale_dim_ == 1) 
        {
          Dtype* scale_diff = scale->mutable_cpu_diff();
          if (scale_param) 
          {
            Dtype result;
            caffe_gpu_dot(outer_dim_, sum_mult, sum_result, &result);
            *scale_diff += result;
          } 
          else 
          {
            caffe_gpu_dot(outer_dim_, sum_mult, sum_result, scale_diff);
          }
        } 
        else 
        {
           //  scale 值来自哪里？
           // 获得 scale  diff 的值
          Dtype* scale_diff = scale->mutable_gpu_diff();
           //   讲多个通道多个样本的梯度进行累加。
           //   最终得到scale的diff 的值，
           //   也就是说，在scale里面并没有去使用
          caffe_gpu_gemv(CblasTrans, outer_dim_, scale_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(scale_param),
                         scale_diff);
        }
      }
    }
  }
  //LOG(INFO)<<"propagate_down[0]"<<propagate_down[0];
  //  1
  if (propagate_down[0]) 
  {
    //  N*C*H*W
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    //  c*1
    const Dtype* scale_data = scale->gpu_data();
    //  
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    //  对应位置乘以 scale 因子就可以了。
    //  这里面很明确的
    ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, scale_data, scale_dim_, inner_dim_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);

}  // namespace caffe
