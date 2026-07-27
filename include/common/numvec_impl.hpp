/// @date 2010-09-27
#pragma once
#include  "environment.hpp"
#include  "numvec_decl.hpp"
namespace isdf{

// Templated form of numerical vectors
//
// The main advantage of this portable NumVec structure is that it can
// either own (owndata == true) or view (owndata == false) a piece of
// data.


template <class F> 
  inline NumVec<F>::NumVec    ( Int m ) : m_(m), owndata_(true)
  {
    if(m_>0) { 
      data_ = new F[m_]; 
      if( data_ == NULL ){
        ErrorHandling("Cannot allocate memory.");
      }
    } 
    else 
      data_=NULL;
  }         // -----  end of method NumVec<F>::NumVec  ----- 

template <class F> 
  inline NumVec<F>::NumVec    ( Int m, bool owndata, F* data ) : m_(m), owndata_(owndata)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = new F[m_]; 
        if( data_ == NULL ){
          ErrorHandling("Cannot allocate memory.");
        }
      }
      else
        data_ = NULL;

      if( m_ > 0 ) {
        for( Int i = 0; i < m_; i++ ){
          data_[i] = data[i];
        }
      }
    }
    else{
      data_ = data;
    }
  }         // -----  end of method NumVec<F>::NumVec  ----- 

template <class F> 
  inline NumVec<F>::NumVec    ( const NumVec<F>& C ) : m_(C.m_), owndata_(C.owndata_)
  {
    if( owndata_ ){
      if( m_ > 0 ) { 
        data_ = new F[m_]; 
        if( data_ == NULL ){
          ErrorHandling("Cannot allocate memory.");
        }
      }
      else
        data_ = NULL;

      if( m_ > 0 ) {
        for( Int i = 0; i < m_; i++ ){
          data_[i] = C.data_[i];
        }
      }
    }
    else{
      data_ = C.data_;
    }
  }         // -----  end of method NumVec<F>::NumVec  ----- 

  template <class F>
  inline void NumVec<F>::FreeData() {
          if (data_ != NULL && owndata_) {
          delete[] data_;    // 释放动态分配的内存
          data_ = NULL;      // 将指针设置为 NULL，避免悬挂指针
          m_ = 0;            // 将矩阵的维度设置为 0
          owndata_ = false;  // 标记数据已不再由此对象管理
          }
  }
template < class F > 
  inline NumVec<F>::~NumVec    (  )
  {
    if( owndata_ ){
      if( m_ > 0 ){
        delete[] data_;  
        data_ = NULL;
      }
    }

  }         // -----  end of method NumVec<F>::~NumVec  ----- 


template < class F > 
  inline NumVec<F>& NumVec<F>::operator =    ( const NumVec& C  )
  {
    // Do not copy if it is the same matrix.
    if(C.data_ != data_){
      if( owndata_ ){
        if( m_ > 0 ){
          delete[]  data_;
          data_ = NULL;
        }
      }
      m_ = C.m_;
      owndata_ = C.owndata_;

      if( owndata_ ) {
        if( m_ > 0 ){
          data_ = new F[m_];
          if( data_ == NULL ){
            ErrorHandling("Cannot allocate memory.");
          }
        }
        else{
          data_ = NULL;
        }

        if( m_ > 0 ){
          for( Int i = 0; i < m_; i++ ){
            data_[i] = C.data_[i];
          }
        }
      }
      else{
        data_ = C.data_;
      }
    }


    return *this;
  }         // -----  end of method NumVec<F>::operator=  ----- 


template < class F > 
  inline void NumVec<F>::Resize    ( const Int m )
  {
    if( owndata_ == false ){
      ErrorHandling("Vector being resized must own data.");
    }
    if( m != m_ ){
      if( m_ > 0 ){
        delete[] data_;
        data_ = NULL;
      }
      m_ = m;
      if( m_ > 0 ){
        data_ = new F[m_];
        if( data_ == NULL ){
          ErrorHandling("Cannot allocate memory.");
        }
      }
    }

    return ;
  }         // -----  end of method NumVec<F>::Resize  ----- 


template <class F> 
  inline F& NumVec<F>::operator()    ( Int i )
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];

  }         // -----  end of method NumVec<F>::operator()  ----- 


template <class F>
  inline const F& NumVec<F>::operator()    ( Int i ) const
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];

  }         // -----  end of method NumVec<F>::operator()  ----- 


template <class F> 
  inline F& NumVec<F>::operator[]    ( Int i )
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];

  }         // -----  end of method NumVec<F>::operator[]  ----- 


template <class F> 
  inline const F& NumVec<F>::operator[]    ( Int i ) const
  {
#if ( _DEBUGlevel_ >= 1 )
    if( i < 0 || i >= m_ ){
      std::ostringstream msg;
      msg 
        << "Index is out of bound."  << std::endl
        << "Index bound    ~ (" << m_ << ")" << std::endl
        << "This index     ~ (" << i << ")" << std::endl;
      ErrorHandling(msg.str().c_str());
    }
#endif
    return data_[i];

  }         // -----  end of method NumVec<F>::operator[]  ----- 

// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue( NumVec<F>& vec, F val )
{
  for(Int i=0; i<vec.m(); i++)
    vec(i) = val;
}

template <class F> inline Real Energy( const NumVec<F>& vec )
{
  Real sum = 0;
  for(Int i=0; i<vec.m(); i++)
    sum += std::abs(vec(i)*vec(i));
  return sum;
}

template <class F> inline Real findMin( const NumVec<F>& vec )
{
  Real min = 0.0;
  for(Int i=0; i<vec.m(); i++)
    if(vec(i) < min)
	    min = vec(i);
  return min;
}  

template <class F> inline Real findMax( const NumVec<F>& vec )
{
  Real max = 0.0;
  for(Int i=0; i<vec.m(); i++)
    if(vec(i) > max)
	    max = vec(i);
  return max;
}  



template <class F> inline void Sort( NumVec<F>& vec ){
  std::vector<F>  tvec(vec.m());
  std::copy( vec.Data(), vec.Data() + vec.m(), tvec.begin() );
  std::sort( tvec.begin(), tvec.end() );
  for(Int i = 0; i < vec.m(); i++){
    vec(i) = tvec[i];
  }
  return;
}

template <class F>
void printCpxV(NumVec<F> &Mat, std::ostream &out) {
    Int r1 = Mat.m();
    out << "test" << std::endl;
    for (Int i = 0; i < r1; i++) {
        out << Mat(i) << std::endl;
    }
    out.flush();
}



} // namespace isdf

