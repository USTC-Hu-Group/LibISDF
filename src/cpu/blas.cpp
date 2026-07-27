#include "../../include/cpu/blas.hpp"

namespace isdf {
namespace blas {

extern "C" {
   void BLAS(dscal)
   ( const Int* n, const double* alpha, double* x, const Int* incx );
   void BLAS(zscal)
   ( const Int* n, const dcomplex* alpha, dcomplex* x,const Int* incx );
    void BLAS(dcopy)
   ( const Int* n, const double* x, const Int* incx, double* y, const Int* incy );
   void BLAS(zcopy)
   ( const Int* n, const dcomplex* x, const Int* incx,dcomplex* y, const Int* incy );

  void BLAS(dgemm)
    ( const char* transA, const char* transB,
      const Int* m, const Int* n, const Int* k,
      const double* alpha, const double* A, const Int* lda,
      const double* B, const Int* ldb,
      const double* beta,        double* C, const Int* ldc );
  void BLAS(zgemm)
     ( const char* transA, const char* transB,
       const Int* m, const Int* n, const Int* k,
       const dcomplex* alpha,const dcomplex* A, const Int* lda,
       const dcomplex* B, const Int* ldb,const dcomplex* beta,
     dcomplex* C, const Int* ldc );

  void BLAS(dtrsm)
    ( const char* side, const char* uplo, const char* transA, const char* diag,
      const Int* m, const Int* n,
      const double* alpha, const double* A, const Int* lda,
      double* B, const Int* ldb );

} 
void Scal( Int n, double alpha, double* x, Int incx )
{ BLAS(dscal)( &n, &alpha, x, &incx ); }

void Scal( Int n, dcomplex alpha, dcomplex* x, Int incx )
{ BLAS(zscal)( &n, &alpha, x, &incx ); }

void Copy( Int n, const double* x, Int incx, double* y, Int incy )
{ BLAS(dcopy)( &n, x, &incx, y, &incy ); }

void Copy( Int n, const dcomplex* x, Int incx, dcomplex* y, Int incy )
{ BLAS(zcopy)( &n, x, &incx, y, &incy ); }

void Gemm
( char transA, char transB,
  Int m, Int n, Int k, 
  double alpha, const double* A, Int lda, const double* B, Int ldb,
  double beta,        double* C, Int ldc )
{
  const char fixedTransA = ( transA == 'C' ? 'T' : transA );
  const char fixedTransB = ( transB == 'C' ? 'T' : transB );
  BLAS(dgemm)( &fixedTransA, &fixedTransB, &m, &n, &k,
      &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}
void Gemm
( char transA, char transB, Int m, Int n, Int k,
    dcomplex alpha, const dcomplex* A, Int lda, const dcomplex* B, Int ldb,
      dcomplex beta,        dcomplex* C, Int ldc )
{
	  BLAS(zgemm)( &transA, &transB, &m, &n, &k,
			        &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

void Trsm
( char side, char uplo, char trans, char unit, Int m, Int n,
  double alpha, const double* A, Int lda, double* B, Int ldb )
{
  const char fixedTrans = ( trans == 'C' ? 'T' : trans );
  BLAS(dtrsm)( &side, &uplo, &fixedTrans, &unit, &m, &n,
      &alpha, A, &lda, B, &ldb );
} 


} // namespace blas
} // namespace isdf
