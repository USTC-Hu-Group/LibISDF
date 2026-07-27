/// @file numvec_decl.hpp
/// @brief  Numerical vector.
/// @date 2010-09-27
#pragma once
#include "environment.hpp"
namespace  isdf{


template <class F> class NumVec
{
public:
  Int  m_;                                
  bool owndata_;                          
  F* data_;                              
public:
  NumVec(Int m = 0);
  NumVec(Int m, bool owndata, F* data);
  NumVec(const NumVec& C);
  void FreeData() ;
  ~NumVec();

  NumVec& operator=(const NumVec& C);

  void Resize ( Int m );

  const F& operator()(Int i) const;  
  F& operator()(Int i);  
  const F& operator[](Int i) const;
  F& operator[](Int i);

  bool IsOwnData() const { return owndata_; }

  F*   Data() const { return data_; }

  Int  m () const { return m_; }

  Int Size() const { return m_; }
};

typedef NumVec<bool>       BolNumVec;
typedef NumVec<Int>        IntNumVec;
typedef NumVec<Real>       DblNumVec;
typedef NumVec<Complex>    CpxNumVec;
typedef NumVec<float>      FloNumVec;


template <class F> inline void SetValue( NumVec<F>& vec, F val );
template <class F> inline Real Energy( const NumVec<F>& vec );
template <class F> inline Real findMin( const NumVec<F>& vec );
template <class F> inline Real findMax( const NumVec<F>& vec );
template <class F> inline void Sort( NumVec<F>& vec );
template <class F> void printCpxV(NumVec<F> &Mat, std::ostream &out);
} // namespace isdf

//#include "numvec_impl.hpp"

