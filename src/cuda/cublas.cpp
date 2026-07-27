#ifdef GPU   
#include  "../../include/common/environment.hpp"

#include "../../include/cuda/cublas.hpp"

inline void __cublas_error(cublasStatus_t status, const char *file, int line, const char *msg)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
      float* foo = NULL;
      float bar = foo[0];
      printf("Tried to segfault! %f\n", bar);

        printf("\nCUBLAS Error in %s, line %d: %s\n %s\n", file, line, cublasGetStatusString(status), msg);
        cudaDeviceReset();
        exit(-1);
    }
}

#define CUBLAS_ERROR(status, msg) __cublas_error( status, __FILE__, __LINE__, msg )

namespace isdf {
cublasHandle_t hcublas;

namespace cublas {

typedef  int               Int;
typedef  cuComplex         scomplex;

void Init(void)
{
    CUBLAS_ERROR( cublasCreate(&hcublas), "Failed to initialze CUBLAS!" );
}

void Destroy(void)
{
    CUBLAS_ERROR( cublasDestroy(hcublas), "Failed to initialze CUBLAS!" );
}

 void Scal (int n, const double *alpha, double *x, int incx)
{
    CUBLAS_ERROR( cublasDscal(hcublas, n, alpha, x, incx), "cublas Dscal failed! ");
    return;
}


 void Gemm 
           ( cublasOperation_t transA, cublasOperation_t transB, Int m, Int n,const Int k,
            const double *alpha, const double* A, Int lda, const double* B, const Int ldb,
             const double *beta,        double* C, Int ldc )
{
    CUBLAS_ERROR(cublasDgemm_v2(hcublas, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), " cublasDgemm failed !");
    return;
}


 void Trsm ( cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, 
             cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, 
             int lda, double *B, int ldb )
{
    CUBLAS_ERROR( cublasDtrsm(hcublas, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb ), 
                  " cublas Dtrsm failed! ");
    return;
}

} // namespace cublas
} // namespace isdf

#endif
