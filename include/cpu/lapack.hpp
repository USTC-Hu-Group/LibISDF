#pragma once
#include  "../common/environment.hpp"

namespace isdf {

namespace lapack {
typedef  int                    Int; 
typedef  std::complex<float>    scomplex;
typedef  std::complex<double>   dcomplex;
void Potrf( char uplo, Int n, const float* A, Int lda );
void Potrf( char uplo, Int n, const double* A, Int lda );
void Potrf( char uplo, Int n, const scomplex* A, Int lda );
void Potrf( char uplo, Int n, const dcomplex* A, Int lda );
void Orth( Int m, Int n, double* A, Int lda );
void Potrf( char uplo, Int n, const double* A, Int lda );

void Lacpy( char uplo, Int m, Int n, const double* A, Int lda,
    double* B, Int ldb    );
void Lacpy( char uplo, Int m, Int n, const dcomplex* A, Int lda,
    dcomplex* B, Int ldb    );

} // namespace lapack
} // namespace isdf
