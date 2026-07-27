#include "../../include/cpu/lapack.hpp"
namespace isdf {
namespace lapack {

extern "C" {


  // Copy

  void LAPACK(dlacpy)
    ( const char* uplo, const Int* m, const Int* n, 
      const double* A, const Int *lda, 
      double* B, const Int *ldb );
  void LAPACK(zlacpy)
    ( const char* uplo, const Int* m, const Int* n, 
      const dcomplex* A, const Int *lda, 
      dcomplex* B, const Int *ldb );
  void LAPACK(spotrf)
    ( const char* uplo, const Int* n, const float* A, const Int* lda,
		        Int* info );
  void LAPACK(dpotrf)
    ( const char* uplo, const Int* n, const double* A, const Int* lda,
		          Int* info );
  void LAPACK(cpotrf)
    ( const char* uplo, const Int* n, const scomplex* A,
		            const Int* lda, Int* info );
  void LAPACK(zpotrf)
    ( const char* uplo, const Int* n, const dcomplex* A,
			    const Int* lda, Int* info );
  void LAPACK(dgesvd)( const char* jobu, const char* jobvt, const Int* m, const Int* n, double* A, const Int* lda, double* s, double* U, const Int* ldu, double* VTrans, const Int* ldvt, double* work, const Int* lwork, Int* info );
} // extern "C"


// *********************************************************************
// Cholesky factorization
// *********************************************************************

void Potrf( char uplo, Int n, const float* A, Int lda )
{
  Int info;
  LAPACK(spotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "spotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is not HPD.");
}

void Potrf( char uplo, Int n, const double* A, Int lda )
{
  Int info;
  LAPACK(dpotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "dpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 ){
    std::ostringstream msg;
    msg << "dpotrf returned with info = " << info << std::endl;
    msg << "A(info,info) = " << A[info-1+(info-1)*lda] << std::endl;
    ErrorHandling( msg.str().c_str() );
  }
}

void Potrf( char uplo, Int n, const scomplex* A, Int lda )
{
  Int info;
  LAPACK(cpotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "cpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is not HPD.");
}

void Potrf( char uplo, Int n, const dcomplex* A, Int lda )
{
  Int info;
  LAPACK(zpotrf)( &uplo, &n, A, &lda, &info );
  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "zpotrf returned with info = " << info;
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
    ErrorHandling("Matrix is not HPD.");
}


// *********************************************************************
// Copy
// *********************************************************************

void Lacpy( char uplo, Int m, Int n, const double* A, Int lda,
    double* B, Int ldb    ){
  LAPACK(dlacpy)( &uplo, &m, &n, A, &lda, B, &ldb );
}

void Lacpy( char uplo, Int m, Int n, const dcomplex* A, Int lda,
    dcomplex* B, Int ldb    ){
  LAPACK(zlacpy)( &uplo, &m, &n, A, &lda, B, &ldb );
}



void Orth( Int m, Int n, double* A, Int lda ){
  if( m==0 || n==0 )
    return;

  // Overwrite mode
  const char jobu='O', jobvt='N';
  std::vector<double> s(n);
  Int lwork=-1, info;
  double dummyWork;
  double dummyReal;
  Int dummyInt = 1;

  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, &s[0], &dummyReal, &dummyInt,
      &dummyReal, &dummyInt, &dummyWork, &lwork, &info );

  lwork = dummyWork;
  std::vector<double> work(lwork);
  LAPACK(dgesvd)
    ( &jobu, &jobvt, &m, &n, A, &lda, &s[0], &dummyReal, &dummyInt,
      &dummyReal, &dummyInt, &work[0], &lwork, &info );
  if( info < 0 )
  { 
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }
  else if( info > 0 )
  {
    ErrorHandling("dgesvd's updating process failed");
  }
  return ;
}



} // namespace lapack
} // namespace isdf
