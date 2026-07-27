#ifdef GPU

#pragma once
#include "../common/environment.hpp"
#include "../common/nummat_decl.hpp"
#include "cuda_utils.h"

namespace  isdf{

template <class F>
  class cuNumMat
  {
  public:
    Int m_, n_;
    bool owndata_;
    F* data_;

  public:

    cuNumMat(Int m=0, Int n=0);

    cuNumMat(Int m, Int n, bool owndata, F* data);

    cuNumMat(const cuNumMat& C); 

    ~cuNumMat();

    cuNumMat& operator=(const cuNumMat& C);

    void Resize(Int m, Int n);

    const F& operator()(Int i, Int j) const;  

    F& operator()(Int i, Int j);  

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* VecData(Int j)  const; 

    Int m() const { return m_; }

    Int n() const { return n_; }

    Int Size() const { return m_ * n_; }

    void CopyTo(NumMat<F> & C); 

    void CopyFrom(const NumMat<F> &C);

    void CopyTo(cuNumMat<F> & C); 

    void CopyFrom(const cuNumMat<F> &C);

    void FreeData();
  };

typedef cuNumMat<bool>     cuBolNumMat;
typedef cuNumMat<Int>      cuIntNumMat;
typedef cuNumMat<Real>     cuDblNumMat;
typedef cuNumMat<float>    cuFltNumMat;
typedef cuNumMat<Complex>  cuCpxNumMat;

template <typename F>
void printCpxM(F &Mat, std::ostream &out);

} // namespace isdf

#endif // GPU

