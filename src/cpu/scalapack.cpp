#include "../../include/cpu/scalapack.hpp"

namespace isdf {
namespace scalapack {


void
Gemm( char transA, char transB,
    Int m, Int n, Int k,
    double alpha,
    double* A, Int ia, Int ja, Int* desca, 
    double* B, Int ib, Int jb, Int* descb,
    double beta,
    double* C, Int ic, Int jc, Int* descc,
    Int contxt){


  SCALAPACK(pdgemm)( &transA, &transB,
      &m, &n, &k,
      &alpha,
      A, &ia, &ja, desca, 
      B, &ib, &jb, descb,
      &beta,
      C, &ic, &jc, descc);

  return;
}





void QRCPF( Int m, Int n, double* A, Int* desca, Int* piv, double* tau) 
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int lwork=-1, info;
  double dummyWork;
  int I_ONE = 1;

  SCALAPACK(pdgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &dummyWork, &lwork, &info);

  lwork = dummyWork;
  std::vector<double> work(lwork);
  SCALAPACK(pdgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &work[0], &lwork, &info);

  for( Int i = 0; i < n; i++ ){
    piv[i]--;
  }

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
    ErrorHandling( msg.str().c_str() );
  }

  return;
}







} // namespace scalapack
} // namespace isdf
