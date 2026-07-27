#ifdef  GPU
#include "../../include/cuda/cusolver.hpp"



namespace isdf {
cusolverDnHandle_t cusolverH;
namespace cusolver {
/*
void Init(void)
{
  cusolverH = NULL;
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;


  int rank = 0, ndev = 0, curdev = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaError_t ec = cudaGetDeviceCount(&ndev);
  cudaError_t eg = cudaGetDevice(&curdev);
  fprintf(stderr,
      "[cusolver::Init] rank=%d  device_count=%d (%s)  current_device=%d (%s)\n",
      rank, ndev, cudaGetErrorString(ec), curdev, cudaGetErrorString(eg));
  fflush(stderr);

  if (ndev > 0) {
    cudaSetDevice(rank % ndev);
    cudaGetDevice(&curdev);
    fprintf(stderr, "[cusolver::Init] rank=%d after setDevice -> device=%d\n",
            rank, curdev);
    fflush(stderr);
  }

  status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == status);


  if( status != CUSOLVER_STATUS_SUCCESS ) {
    std::ostringstream msg;
    msg << " CU_SOLVER init Error... " << std::endl;
  }
  else{
    isdfOFS << " cuSparse Solver is initialized successfully " << std::endl;
  }
   
}
*/

void Init(void)
{
  cusolverH = NULL;

  int rank = -1;
  int inited = 0;
  MPI_Initialized(&inited);
  if (inited) MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  cudaError_t sync_e = cudaDeviceSynchronize();
  cudaError_t last_e = cudaGetLastError();

  int dev = -1;
  cudaGetDevice(&dev);

  fprintf(stderr,
      "[cusolver::Init] rank=%d device=%d  pre_sync=%d (%s)  pre_last=%d (%s)\n",
      rank, dev,
      (int)sync_e, cudaGetErrorString(sync_e),
      (int)last_e, cudaGetErrorString(last_e));
  fflush(stderr);

  cusolverStatus_t status = cusolverDnCreate(&cusolverH);

  if (status != CUSOLVER_STATUS_SUCCESS) {
    cudaError_t post_e = cudaGetLastError();
    fprintf(stderr,
        "[cusolver::Init] FAILED rank=%d device=%d  status=%d  post_cuda=%d (%s)\n",
        rank, dev, (int)status,
        (int)post_e, cudaGetErrorString(post_e));
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  isdfOFS << " cuSolver initialized successfully on device " << dev
          << " (rank=" << rank << ")" << std::endl;
}


void Destroy(void)
{
  if (cusolverH) cusolverDnDestroy(cusolverH);
}

// *********************************************************************
// Cholesky factorization
// *********************************************************************
void Potrf( char uplo_host, Int n, double* A, Int lda ) { 


  int lwork;
  int *info;
  double * work_array;
  cublasFillMode_t uplo; 
  cusolverStatus_t cusolver_status;

  if(uplo_host == 'u' || uplo_host == 'U')
    uplo  = CUBLAS_FILL_MODE_UPPER;
  else 
    uplo = CUBLAS_FILL_MODE_LOWER;

  assert(cudaDeviceSynchronize() == cudaSuccess);

  cusolver_status = 
  cusolverDnDpotrf_bufferSize( cusolverH, 
                               uplo, 
                               n, 
                               A, 
                               lda, 
                               &lwork );
  
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(cudaMalloc( (void**) & work_array, sizeof(double) * lwork) == cudaSuccess); 
  assert(cudaMalloc( (void**) & info, sizeof(int)  ) == cudaSuccess); 
  cusolver_status =
  cusolverDnDpotrf(cusolverH,
                   uplo,
                   n,
                   A,
                   lda,
                   work_array,
                   lwork,
                   info);

  int info1;
  assert(cudaMemcpy( &info1, info, sizeof(int) , cudaMemcpyDeviceToHost) == cudaSuccess);
  if(info1 != 0) {
    std::ostringstream msg;
    msg << "cu_solver potrf returned with info = " << info1;
    ErrorHandling( msg.str().c_str() );
  }
  assert( cudaSuccess ==cudaDeviceSynchronize());
  assert( cudaSuccess == cudaFree(work_array));
  assert( cudaSuccess == cudaFree(info));
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

}


} // namespace cuSolver
} // namespace isdf
#endif
