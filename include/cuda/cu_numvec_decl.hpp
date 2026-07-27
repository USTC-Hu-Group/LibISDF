#ifdef GPU
#pragma once
#include "../common/environment.hpp"
#include "cuda_utils.h"

namespace  isdf{

template <class F> class cuNumVec
{
public:
  Int  m_;                                
  bool owndata_;                         
  F* data_;                               
  using value_type = F;
public:
  cuNumVec(Int m = 0);
  cuNumVec(Int m, bool owndata, F* data);
  cuNumVec(const cuNumVec& C);
  ~cuNumVec();

  cuNumVec& operator=(const cuNumVec& C);

  void Resize ( Int m );

  const F& operator()(Int i) const;  
  F& operator()(Int i);  
  const F& operator[](Int i) const;
  F& operator[](Int i);

  bool IsOwnData() const { return owndata_; }

  F*   Data() const { return data_; }

  Int  m () const { return m_; }

  Int Size() const { return m_; }

  void CopyTo(NumVec<F> & C); 

  void CopyFrom(const NumVec<F> &C);

  void FreeData();

};

typedef cuNumVec<bool>       cuBolNumVec;
typedef cuNumVec<Int>        cuIntNumVec;
typedef cuNumVec<Real>       cuDblNumVec;
typedef cuNumVec<cuDoubleComplex>    cuCpxNumVec;
template <typename F>
void printCpxV(F &Vec, std::ostream &out);

} // namespace isdf

#endif // GPU
