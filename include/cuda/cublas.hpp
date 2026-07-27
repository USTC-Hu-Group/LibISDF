#ifdef GPU  

#pragma once
#include  "../common/environment.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"
#include "cuda_errors.hpp"

namespace isdf {
extern cublasHandle_t hcublas;
namespace cublas {

typedef  int               Int;
typedef  cuComplex         scomplex;
typedef  cuDoubleComplex   dcomplex;

void Init(void);

void Destroy(void);

 void Gemm 
           ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n, Int k,
            const double *alpha, const double* A, Int lda, const double* B, Int ldb,
            const double *beta,              double* C, Int ldc );
 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, 
             int lda, double *B, int ldb );
} // namespace cublas
} // namespace isdf

#endif
