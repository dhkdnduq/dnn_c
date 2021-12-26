#pragma once
#include "dnn_module_torch.h"
#include "dnn_module_tensorrt.h"
class dnn_impl;
class dnn_module_tensorrt;
class dnn_module_torch;

template < class T>
class dnn_base {
 private:

 public:
  dnn_base<T>() { 
  }
 
  explicit dnn_base(unique_ptr<T> tensor_impl)
      : impl_(std::move(tensor_impl)){
    if (impl_.get() == nullptr)
      throw std::runtime_error("dnnimpl with nullptr is not supported");
    };
  dnn_base(const dnn_base&) = default;
  dnn_base(dnn_base&&) = default;
  void reset() {
    impl_.reset();
  }
  void init() { 
    impl_.reset(new T());
  }
  dnn_base& operator=(const dnn_base& x)& {
    impl_ = x.impl_;
    return *this;
  }
  dnn_base& operator=(dnn_base&& x) & {
    impl_ =std::move( x.impl_);
    return *this;
  }
  dnn_base& operator=(const dnn_base&&) && = delete;
  dnn_base& operator=(dnn_base&&) && = delete;

  T* func() const {
    return this->impl_.get();
  }
 
  unique_ptr<T> impl_;

  
};