#ifdef GPU
#pragma once
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuComplex.h"
#define NSTREAM 1

//Macros for n-dimensional array access (column major format)
#define DIM2(x, y, xdim)                                       ((y * xdim) + x)
#define DIM3(x, y, z, xdim, ydim)                              ((((z * ydim) + y) * xdim) + x)
#define DIM4(x1, x2, x3, x4, x1dim, x2dim, x3dim)              ((((((x4 * x3dim) + x3) * x2dim) + x2) * x1dim) + x2)
#define DIM5(x1, x2, x3, x4, x5, x1dim, x2dim, x3dim, x4dim)   ((((((((x5 * x4dim) + x4) * x3dim) + x3) * x2dim) + x2) * x1dim) + x1)

#define CUDA_CALL(function) {\
cudaError_t err = function; \
if (err != cudaSuccess) \
  fprintf(stderr, "CURROR [%s,%d]: %s \n", \
  __FILE__,  __LINE__, cudaGetErrorString(err)); \
}

#define CUDA_FFT_CALL(function) {\
cufftResult err = function; \
if (err != CUFFT_SUCCESS) \
  fprintf(stderr, "CURROR [%s,%d]: %d \n", \
  __FILE__,  __LINE__, err); \
}
namespace isdf {
extern double * dev_vtot;
extern double * dev_gkkR2C;
extern int    * dev_idxFineGridR2C;

extern int    * dev_NLindex;
extern int    * dev_NLpart;
extern double * dev_NLvecFine;
extern double * dev_atom_weight;
extern double * dev_temp_weight;
extern cuDoubleComplex * dev_temp_weight_complex;
extern double * dev_TeterPrecond;
extern int totPart_gpu;

extern bool vtot_gpu_flag;
extern bool NL_gpu_flag;
extern bool teter_gpu_flag;

void cuda_setValue( double* dev, double val, int len );
void cuda_memcpy_GPU2CPU( void *cpu, void * gpu, size_t size );
void cuda_memcpy_CPU2GPU( void *gpu, void * cpu, size_t size );
void cuda_memcpy_GPU2GPU( void *dest, void * src, size_t size );
void cuda_free( void *ptr);
void *cuda_malloc( size_t size);
void cuda_memory(void);
//void getCuvreal(double *psiCol,int *pivQR,double* phiMuv,int rk,int ntot,int num_local,int idx);
void getelemproduct(int rk1, int rk2, double* C, const double* A, const double* B);
//void calculatekmeansweight(double *psi,int cols1,double*phi, int cols2,double* weightLocal,int rows);
void calculateKmeansWeight(const double* psi, int rows, int i0_1, int i1_1,int i0_2, int i1_2, double* weightLocal, cudaStream_t stream = 0);

void pdist2_GPU(double* A, double* B, double* DD, int na, int nb);
//void findMin_GPU(double* A,int col,int* Imin,int m, int n);
void findMin_GPU(double* A,int* Imin,double* amin,int col,int m, int n, bool computeMin);
void getpsiMuT(double *psiCol, int *pivQR, double* phiMuv, int rk, int ntot, int num_local, int idx);
void cuda_gather_rows(const double* d_src, const int* d_piv,int rk, int ncol, int ldSrc, double* d_dst);
void cuda_add_diag(double* d_A, int n, double eps);

void cuda_sync();
void cuda_scatter_cols(const double* d_src, const int* d_ownedNu, int rk, int selectedRow, double* d_dst);
}


void cuda_cal_recvk_isdf( int * recvk, int * recvdisp, int width, int heightLocal, int mpisize);
void cuda_cal_sendk_isdf( int * sendk, int * senddispl, int widthLocal, int height, int heightBlockSize, int mpisize);
void cuda_mapping_to_buf_isdf( double * buf, double * psi, int * index, int len );
void cuda_mapping_from_buf_isdf( double * psi, double * buf, int * index, int len );
#endif
