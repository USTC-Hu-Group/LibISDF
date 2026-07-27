#pragma once
#include  "../common/environment.hpp"
namespace isdf{
namespace scalapack{
typedef  int                    Int;
typedef  std::complex<float>    scomplex;
typedef  std::complex<double>   dcomplex;
extern "C"{
void Cblacs_get(const Int contxt, const Int what, Int* val);
void Cblacs_gridinit(Int* contxt, const char* order, const Int nprow, const Int npcol);
void Cblacs_gridmap(Int* contxt, Int* pmap, const Int ldpmap, const Int nprow, const Int npcol);
void Cblacs_gridinfo(const Int contxt,  Int* nprow, Int* npcol, 
    Int* myprow, Int* mypcol);
void Cblacs_gridexit    (    int contxt );    

int SCALAPACK(numroc)(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

void SCALAPACK(descinit)(Int* desc, const Int* m, const Int * n, const Int* mb,
    const Int* nb, const Int* irsrc, const Int* icsrc,
    const Int* contxt, const Int* lld, Int* info);


void SCALAPACK(pdgemm)(const char* transA, const char* transB,
    const Int* m, const Int* n, const Int* k,
    const double* alpha,
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* B, const Int* ib, const Int* jb, const Int* descb,
    const double* beta,
    double* C, const Int* ic, const Int* jc, const Int* descc);


void SCALAPACK(pdtradd)(const char* uplo, const char* trans, const Int* m, const Int* n,
                        const double* alpha,
                        const double* a, const Int* ia, const Int* ja, const Int* desca, 
                        const double* beta,
                        double* c, const Int* ic, const Int* jc, const Int* descc);


void SCALAPACK(pdgeadd)(const char *trans,
    const Int* m, const Int* n, 
    const double *alpha, 
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* beta,
    double* B, const Int* ib, const Int* jb, const Int* descb);


void SCALAPACK(pdgemr2d)(const Int* m, const Int* n, const double* A, const Int* ia, 
    const Int* ja, const Int* desca, double* B,
    const Int* ib, const Int* jb, const Int* descb,
    const Int* contxt);

void SCALAPACK(pdgeqpf)( Int* m, Int* n, double* A, Int* ia, Int* ja,
    Int* desca, Int* ipiv, double* itau, double* work, Int* lwork, 
    Int* info );

void SCALAPACK(pdpotrf)( const char* uplo, const Int* n, 
    double* A, const Int* ia, const Int* ja, const Int* desca, 
    Int* info );

void SCALAPACK(pdtrsm)( const char* side, const char* uplo, const char* trans, const char* diag, const int* m, const int* n, const double* alpha,const double* a, const int* ia, const int* ja, const int* desca, double* b, const int* ib, const int* jb, const int* descb );



void SCALAPACK(pdpotri)( const char* uplo, const Int* n, 
    double* A, const Int* ia, const Int* ja, const Int* desca, 
    Int* info );



} //extern "C"

void QRCPF( Int m, Int n, double* A, Int* desca, Int* piv, double* tau ); 
} // namespace scalapack
} // namespace isdf

