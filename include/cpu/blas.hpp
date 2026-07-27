#pragma once
#include  "../common/environment.hpp"

namespace isdf {

namespace blas {

typedef  int                    Int;
typedef  std::complex<float>    scomplex;
typedef  std::complex<double>   dcomplex;
void Scal( Int n, double alpha, double* x, Int incx );
void Scal( Int n, dcomplex alpha, dcomplex* x, Int incx );
template<typename F> void Scal( Int n, F alpha, F* x, Int incx );

void Copy( Int n, const double* x, Int incx, double* y, Int incy );
void Copy( Int n, const dcomplex* x, Int incx, dcomplex* y, Int incy );
template<typename T> void Copy( Int n, const T* x, Int incx, T* y, Int incy );


void Gemm
  ( char transA, char transB, Int m, Int n, Int k,
    double alpha, const double* A, Int lda, const double* B, Int ldb,
    double beta,        double* C, Int ldc );

void Gemm
  ( char transA, char transB, Int m, Int n, Int k, dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb, dcomplex beta, dcomplex* C, Int ldc );

void Trsm
  ( char side,  char uplo, char trans, char unit, Int m, Int n,
    double alpha, const double* A, Int lda, double* B, Int ldb );

} // namespace blas
} // namespace isdf

