#ifdef GPU
#pragma once
#include  "cu_numvec_decl.hpp"

namespace  isdf{

template <class F> 
  inline cuNumVec<F>::cuNumVec( Int m ) 
  {
    owndata_ = true;
    if(m>0) { 
      m_ = m;
      data_ = (F*) cuda_malloc( sizeof(F) * m_ );
    } 
    else 
      data_=NULL;
  }         
template <class F> 
  inline cuNumVec<F>::cuNumVec    ( Int m, bool owndata, F* data ) : m_(m), owndata_(owndata)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = (F*) cuda_malloc( sizeof(F) * m_ );
      }
      else
        data_ = NULL;

      if( m_ > 0 ) {
        cuda_memcpy_GPU2GPU(data_, data, sizeof(F)*m_);
      }
    }
    else{
      data_ = data;
    }
  }        

template <class F> 
  inline cuNumVec<F>::cuNumVec    ( const cuNumVec<F>& C ) : m_(C.m_), owndata_(C.owndata_)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = (F*) cuda_malloc( sizeof(F) * m_ );
      }
      else
        data_ = NULL;

      if( m_ > 0 ) {
        cuda_memcpy_GPU2GPU(data_, C.data_, sizeof(F)*m_);
      }
    }
    else{
      data_ = C.data_;
    }
  }          

template < class F > 
  inline cuNumVec<F>::~cuNumVec    (  )
  {
    if( owndata_ ){
      if( m_ > 0 ){
        cuda_free(data_);
        data_ = NULL;
      }
    }

  }          


template < class F > 
  inline cuNumVec<F>& cuNumVec<F>::operator =    ( const cuNumVec& C  )
  {
    if(C.data_ != data_){
      if( owndata_ ){
        if( m_ > 0 ){
          cuda_free(data_);
          data_ = NULL;
        }
      }
      m_ = C.m_;
      owndata_ = C.owndata_;

      if( owndata_ ) {
        if( m_ > 0 ){
          data_ = (F*) cuda_malloc( sizeof(F) * m_ );
        }
        else{
          data_ = NULL;
        }

        if( m_ > 0 ){
          cuda_memcpy_GPU2GPU(data_, C.data_, sizeof(F)*m_);
        }
      }
      else{
        data_ = C.data_;
      }
    }


    return *this;
  }          


template < class F > 
  inline void cuNumVec<F>::Resize    ( const Int m )
  {
    if( owndata_ == false ){
      ErrorHandling("Vector being resized must own data.");
    }
    if( m != m_ ){
      if( m_ > 0 ){
        cuda_free(data_);
        data_ = NULL;
      }
      m_ = m;
      if( m_ > 0 ){
        data_ = (F*) cuda_malloc( sizeof(F) * m_ );
      }
    }

    return ;
  }          


template <class F> 
  inline F& cuNumVec<F>::operator()    ( Int i )
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }          


template <class F>
  inline const F& cuNumVec<F>::operator()    ( Int i ) const
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }          


template <class F> 
  inline F& cuNumVec<F>::operator[]    ( Int i )
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }          


template <class F> 
  inline const F& cuNumVec<F>::operator[]    ( Int i ) const
  {
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
    return data_[i];

  }        

template <class F> inline void cuNumVec<F>::CopyTo(NumVec<F> &C) {
  if( C.m_ < m_ ) 
  { 
    C.Resize(m_);
  }
  if(C.m_ >= m_) {
    if(m_>0 ) { cuda_memcpy_GPU2CPU(C.data_, data_, sizeof(F)*m_);}
  }
}
template <class F> inline void cuNumVec<F>::CopyFrom(const NumVec<F> &C) {
  if( C.m_ > m_ ) 
  { 
    std:: cout << " GPU memory not big enough. " << m_<<" "<< C.m_ << std:: endl;
    cuda_free(data_);
    m_ = C.m_; 
    if(m_>0 ) { data_ = (F*)cuda_malloc( sizeof(F) * m_ ); } else data_=NULL;
   }
  if(C.m_ <= m_) {
    if(m_>0 ) { cuda_memcpy_CPU2GPU(data_, C.data_, sizeof(F)*C.m_);}
  }
}


template <class F> inline void cuNumVec<F>::FreeData() {
  if (data_ != NULL && owndata_) {
    cuda_free(data_);     
    data_ = NULL;         
    m_ = 0;              
    owndata_ = false;    
  }
}

template <class F> inline void SetValue( cuNumVec<F>& vec, F val )
{
  cuda_setValue(vec.data_, val, vec.m_);
}

template <typename F>
void printCpxV(F &Vec, std::ostream &out)
{
  Int r1 = Vec.m();
  out << "test" << std::endl;
  using CPUVecType = typename std::conditional<sizeof(typename F::value_type) == 4, IntNumVec, DblNumVec>::type;
  CPUVecType Vec_CPU(r1);  
  cuda_memcpy_GPU2CPU(Vec_CPU.Data(), Vec.Data(), r1 * sizeof(Vec(0)));
  for (Int i = 0; i < r1; i++)
  {
    out << Vec_CPU(i) << std::endl;
  }
  out.flush();
}

} // namespace isdf

#endif // GPU
