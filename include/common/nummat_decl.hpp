#pragma once
#include "environment.hpp"

namespace  isdf{

template <class F>
  class NumMat
  {
  public:
    Int m_, n_;
    bool owndata_;
    F* data_;
  public:
    NumMat(Int m=0, Int n=0);

    NumMat(Int m, Int n, bool owndata, F* data);

    NumMat(const NumMat& C);
    void FreeData() ;
    void Reset();
    ~NumMat();

    NumMat& operator=(const NumMat& C);

    void Resize(Int m, Int n);

    const F& operator()(Int i, Int j) const;  

    F& operator()(Int i, Int j);  

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* VecData(Int j)  const; 

    Int m() const { return m_; }

    Int n() const { return n_; }
    void RevertRearrange(Int blockNum);
    void ExchangeCols(IntNumVec& colMap);
    void Rearrange(Int blockNum);

    Int Size() const { return m_ * n_; }

  };
typedef NumMat<bool>     BolNumMat;
typedef NumMat<Int>      IntNumMat;
typedef NumMat<Real>     DblNumMat;
typedef NumMat<Complex>  CpxNumMat;

template <class F> inline void SetValue(NumMat<F>& M, F val);
template <class F> inline Real Energy(const NumMat<F>& M);
template <class F> inline void Transpose ( const NumMat<F>& A, NumMat<F>& B );
template <class F> inline void Symmetrize( NumMat<F>& A );
template <class F> void printCpxM(NumMat<F> &Mat, std::ostream &out);


} // namespace isdf

