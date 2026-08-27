#ifdef GPU
#pragma once
#include "cu_nummat.hpp"

namespace isdf{

template <class F> inline cuNumMat<F>::cuNumMat(Int m, Int n): m_(m), n_(n), owndata_(true) {
  if(m_>0 && n_>0) 
     data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); 
  else 
     data_=NULL;
}

template <class F> inline cuNumMat<F>::cuNumMat(Int m, Int n, bool owndata, F* data): m_(m), n_(n), owndata_(owndata) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); }  else data_=NULL;
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU (data_, data, sizeof(F)*m_*n_); }
  } else {
    data_ = data;
  }
}

template <class F> inline cuNumMat<F>::cuNumMat(const cuNumMat& C): m_(C.m_), n_(C.n_), owndata_(C.owndata_) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU (data_, C.data_, sizeof(F)*m_*n_); }
  } else {
    data_ = C.data_;
  }
}

template <class F> inline cuNumMat<F>::~cuNumMat() {
  if(owndata_) {
    if(m_>0 && n_>0) { cuda_free(data_); data_ = NULL; }
  }
}

template <class F> inline cuNumMat<F>& cuNumMat<F>::operator=(const cuNumMat& C) {
  if(C.data_ == data_) return *this;
  if(owndata_) {
    if(m_>0 && n_>0) { cuda_free(data_); data_ = NULL; }
  }
  m_ = C.m_; n_=C.n_; owndata_=C.owndata_;
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU(data_, C.data_, sizeof(F)*m_*n_); }
   // if(m_>0 && n_>0) { for(size_t i=0; i<(size_t)m_*n_; i++) data_[i] = C.data_[i]; }
  } else {
    data_ = C.data_;
  }
  return *this;
}

template <class F> inline void cuNumMat<F>::Resize(Int m, Int n)  {
  if( owndata_ == false ){
    ErrorHandling("Matrix being resized must own data.");
  }
  if(m_!=m || n_!=n) {
    if(m_>0 && n_>0) { cuda_free(data_); data_ = NULL; }
    m_ = m; n_ = n;
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
  }
}

template <class F> inline void cuNumMat<F>::CopyTo(cuNumMat<F> &C) {
  if( (size_t)C.m_*C.n_ < (size_t) m_ * n_) 
  { 
    C.Resize(m_, n_);
  }
  if((size_t)C.m_*C.n_ >= (size_t)m_*n_) {
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU(C.data_, data_, sizeof(F)*m_*n_);}
  }
}

template <class F> inline void cuNumMat<F>::CopyTo(NumMat<F> &C) {
  if( (size_t)C.m_*C.n_ < (size_t)m_ * n_) 
  { 
    C.Resize(m_, n_);
  }
  if((size_t)C.m_*C.n_ >= (size_t)m_*n_) {
    if(m_>0 && n_>0) { cuda_memcpy_GPU2CPU(C.data_, data_, sizeof(F)*m_*n_);}
  }
}

template <class F> inline void cuNumMat<F>::CopyFrom(const cuNumMat<F> &C) {
  if( (size_t)C.m_*C.n_ > (size_t)m_ * n_) 
  { 
    std:: cout << " GPU memory not big enough. " << m_*n_ <<" "<< C.m_ * C.n_ << std:: endl;
    cuda_free(data_);
    m_ = C.m_; n_=C.n_; 
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
   }
  if((size_t)C.m_*C.n_ <= (size_t)m_*n_) {
    if(m_>0 && n_>0) { cuda_memcpy_GPU2GPU(data_, C.data_, sizeof(F)*C.m_*C.n_);}
  }
}

template <class F> inline void cuNumMat<F>::CopyFrom(const NumMat<F> &C) {
  if( (size_t)C.m_*C.n_ > (size_t)m_ * n_) 
  { 
    std:: cout << " GPU memory not big enough. " << m_*n_ <<" "<< C.m_ * C.n_ << std:: endl;
    cuda_free(data_);
    m_ = C.m_; n_=C.n_; 
    if(m_>0 && n_>0) { data_ = (F*)cuda_malloc( sizeof(F) * m_ * n_ ); } else data_=NULL;
   }
  if((size_t)C.m_*C.n_ <= (size_t)m_*n_) {
    if(m_>0 && n_>0) { cuda_memcpy_CPU2GPU(data_, C.data_, sizeof(F)*C.m_*C.n_);}
  }
}



template <class F> 
inline const F& cuNumMat<F>::operator()(Int i, Int j) const  { 
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
  return data_[i+(size_t)j*m_];
}
template <class F>
inline F& cuNumMat<F>::operator()(Int i, Int j)  { 
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
  return data_[i+(size_t)j*m_];
}
template <class F> inline void
cuNumMat<F>::FreeData() {
  if (data_ != NULL && owndata_) {
    cuda_free(data_);    
    data_ = NULL;        
    m_ = 0;              
    n_ = 0;
    owndata_ = false;    
  }
}

template <class F>
inline F* cuNumMat<F>::VecData(Int j)  const 
{ 
  if( j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << n_ << ")" << std::endl
      << "This index     ~ (" << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
  return &(data_[(size_t)j*m_]); 
}

template <typename F>
void printCpxM(F &Mat, std::ostream &out)
{
  Int r1 = Mat.m();
  Int r2 = Mat.n();
  DblNumMat Mat_CPU(r1, r2);
  cuda_memcpy_GPU2CPU(Mat_CPU.Data(), Mat.Data(),sizeof(Mat(0,0)) * r1 * r2);
  out << r1 << std::endl;
  out << r2 << std::endl;
  out<<"GPU"<<std::endl; 

  for (int j = 0; j < r2; j++)
  {
    for (Int i = 0; i < r1; i++)
    {
      out << std::fixed << std::setprecision(20) << Mat_CPU(i, j) << std::endl;
    }
  }
  out.flush();
}


} // namespace isdf

#endif // GPU
