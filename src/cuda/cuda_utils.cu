#include "../../include/cuda/cuda_utils.h"
#include <cfloat>
#define DIM 128
#define LDIM 256
#define LEN 512


__global__ void gpu_setValue_isdf(double *dev, double val, int len)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < len)
        dev[tid] = val;
}

__global__ void elemproduct_isdf(int rk1, int rk2, double* C, const double* A, const double* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rk1 && j < rk2) {
        int index = j * rk1 + i;
        C[index] = A[index] * B[index];
    }
}

__global__ void scatter_cols_kernel(const double* __restrict__ src,
                                    const int*    __restrict__ ownedNu,
                                    int rk, int selectedRow,
                                    double* __restrict__ dst)
{
int mu = blockIdx.y * blockDim.y + threadIdx.y;
int k  = blockIdx.x * blockDim.x + threadIdx.x;
if (mu < rk && k < selectedRow) {
int nu = ownedNu[k];
dst[mu + nu * rk] = src[mu + k * rk];
}
}




__global__ void add_diag_kernel(double* A, int n, double eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i * (n + 1)] += eps;
    }
}




__global__ void gather_rows_kernel(const double* __restrict__ src,
                                   const int*    __restrict__ piv,
                                   int rk, int ncol, int ldSrc,
                                   double* __restrict__ dst)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;   
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    if (j < rk && i < ncol) {
        int srcRow = piv[j];
        dst[j + i * rk] = src[srcRow + i * ldSrc];
    }
}


__global__ void gpu_cal_sendk_isdf( int *sendk, int * senddisps, int widthLocal, int height, int heightBlockSize, int mpisize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < height* widthLocal)
	{
		int i = tid % height;
		int j = tid / height;

		if(height % mpisize == 0){
			sendk[tid] = senddisps[i/heightBlockSize] + j * heightBlockSize + i % heightBlockSize;
		}
		else{
			if( i < ((height%mpisize) * (heightBlockSize + 1)) ) {
				sendk[tid] = senddisps[i/(heightBlockSize + 1)] + j * ( heightBlockSize + 1) + i % ( heightBlockSize + 1);
			}
			else{
				sendk[tid] = senddisps[(height % mpisize) + (i-(height % mpisize)*(heightBlockSize+1))/heightBlockSize]
					+ j * heightBlockSize + (i-(height % mpisize)*(heightBlockSize+1)) % heightBlockSize;
			}
		}
	}
}

__global__ void gpu_cal_recvk_isdf( int *recvk, int * recvdisps, int width, int heightLocal, int mpisize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid < heightLocal* width)
	{
		int i = tid % heightLocal;
		int j = tid / heightLocal;

		recvk[tid] = recvdisps[j%mpisize] + ( j/mpisize) * heightLocal + i;
	}
}

__global__ void gpu_mapping_to_buf_isdf( double *buf, double * psi, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		buf[x] = psi[tid];
	}
}

__global__ void gpu_mapping_from_buf_isdf( double *psi, double * buf, int *index, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x;
	if(tid < len)
	{
		x = index[tid];
		psi[tid] = buf[x];
	}
}


template <int BLOCK_SIZE>
__global__ void kmeansWeightKernel_isdf(const double* __restrict__ psi,
                                        int rows,
                                        int i0_1, int i1_1,
                                        int i0_2, int i1_2,
                                        double* __restrict__ weightLocal)
{
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "BLOCK_SIZE must be a power of two");

    const int kk  = blockIdx.x;
    const int tid = threadIdx.x;

    double s2 = 0.0;
    for (int i = i0_2 + tid; i < i1_2; i += BLOCK_SIZE) {
        double v = psi[(size_t)i * rows + kk];
        s2 += v * v;
    }
    double s1 = 0.0;
    for (int j = i0_1 + tid; j < i1_1; j += BLOCK_SIZE) {
        double v = psi[(size_t)j * rows + kk];
        s1 += v * v;
    }

    __shared__ double s1buf[BLOCK_SIZE];
    __shared__ double s2buf[BLOCK_SIZE];
    s1buf[tid] = s1;
    s2buf[tid] = s2;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s1buf[tid] += s1buf[tid + stride];
            s2buf[tid] += s2buf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        weightLocal[kk] += s1buf[0] * s2buf[0];
    }
}
/*
__global__ void computeDistances_isdf(double* A, double* B, double* DD, int na, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < na && j < nb) {
        double d1 = A[i] - B[j];
        double d2 = A[i + na] - B[j + nb];
        double d3 = A[i + 2*na] - B[j + 2*nb];
        DD[j*na + i] = d1*d1 + d2*d2 + d3*d3;
    }
}
*/
template<int TILE_M, int TILE_N>
__global__ void computeDistances_isdf_tiled(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ DD,
    int na, int nb)
{

    int i0 = blockIdx.x * TILE_M;   //
    int j0 = blockIdx.y * TILE_N;   //

    __shared__ double sA[3][TILE_M];   //
    __shared__ double sB[3][TILE_N];   //

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;


    for (int k = tid; k < TILE_M; k += nthreads) {
        int gi = i0 + k;
        if (gi < na) {
            sA[0][k] = A[gi];
            sA[1][k] = A[gi + na];
            sA[2][k] = A[gi + 2*na];
        } else {
            sA[0][k] = 0.0; sA[1][k] = 0.0; sA[2][k] = 0.0;
        }
    }

    for (int k = tid; k < TILE_N; k += nthreads) {
        int gj = j0 + k;
        if (gj < nb) {
            sB[0][k] = B[gj];
            sB[1][k] = B[gj + nb];
            sB[2][k] = B[gj + 2*nb];
        } else {
            sB[0][k] = 0.0; sB[1][k] = 0.0; sB[2][k] = 0.0;
        }
    }
    __syncthreads();



    int li = threadIdx.x;   
    int lj = threadIdx.y;
    int gi = i0 + li;
    int gj = j0 + lj;

    if (gi < na && gj < nb) {
        double d1 = sA[0][li] - sB[0][lj];
        double d2 = sA[1][li] - sB[1][lj];
        double d3 = sA[2][li] - sB[2][lj];
        DD[(size_t)gj * na + gi] = d1*d1 + d2*d2 + d3*d3;
    }
}

void cuda_cal_sendk_isdf( int * sendk, int * senddispl, int widthLocal, int height, int heightBlockSize, int mpisize)
{
	int total = widthLocal * height;
	int dim = (total + LEN - 1) / LEN;

       	gpu_cal_sendk_isdf<<< dim, LEN>>> ( sendk, senddispl, widthLocal, height, heightBlockSize, mpisize );
#ifdef SYNC
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_cal_recvk_isdf( int * recvk, int * recvdisp, int width, int heightLocal, int mpisize)
{
	int total = width * heightLocal;
	int dim = ( total + LEN - 1 ) / LEN;

	gpu_cal_recvk_isdf<<< dim, LEN>>> ( recvk, recvdisp, width, heightLocal, mpisize );
#ifdef SYNC
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_mapping_from_buf_isdf( double * psi, double * buf, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_from_buf_isdf<<< ndim, DIM>>>( psi, buf, index, len);
#ifdef SYNC
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaDeviceSynchronize() == cudaSuccess );
#endif
}

void cuda_mapping_to_buf_isdf( double * buf, double * psi, int * index, int len )
{
	int ndim = (len + DIM - 1) / DIM;
	gpu_mapping_to_buf_isdf<<< ndim, DIM>>>( buf, psi, index, len);
#ifdef SYNC
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	assert(cudaDeviceSynchronize() == cudaSuccess );
#endif
}


/*
__global__ void findminIndex_col_isdf(double* A, int* Imin, double* amin, int m, int n, bool computeMin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double minValue = A[i];
        int minIndex = 0;
        for (int j = 1; j < n; j++) {
            if (A[j * m + i] < minValue) {
                minValue = A[j * m + i];
                minIndex = j;
            }
        }
        Imin[i] = minIndex;
        if (computeMin) {
            amin[i] = minValue;
        }
    }
}

__global__ void findminIndex_row_isdf(double* A, int* Imin, double* amin, int m, int n, bool computeMin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double minValue = A[i * m];
        int minIndex = 0;
        for (int j = 1; j < m; j++) {
            if (A[i * m + j] < minValue) {
                minValue = A[i * m + j];
                minIndex = j;
            }
        }
        Imin[i] = minIndex;
        if (computeMin) {
            amin[i] = minValue;
        }
    }
}
*/
/*
__global__ void findminIndex_col_isdf(const double* __restrict__ A,
                                      int* Imin, double* amin,
                                      int m, int n, bool computeMin)
{
    int row = blockIdx.x;
    if (row >= m) return;

    constexpr int BS = 256;
    __shared__ double sval[BS];
    __shared__ int    sidx[BS];

    int tid = threadIdx.x;
    double myVal = DBL_MAX;
    int    myIdx = 0;
        for (int j = tid; j < n; j += BS) {
        double v = A[(size_t)j * m + row];
        if (v < myVal) { myVal = v; myIdx = j; }
    }
    sval[tid] = myVal;
    sidx[tid] = myIdx;
    __syncthreads();
       #pragma unroll
    for (int s = BS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double vo = sval[tid + s];
            double vt = sval[tid];
            int    io = sidx[tid + s];
            int    it = sidx[tid];
            if (vo < vt || (vo == vt && io < it)) {
                sval[tid] = vo;
                sidx[tid] = io;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        Imin[row] = sidx[0];
        if (computeMin) amin[row] = sval[0];
    }
}
*/



__global__ void findminIndex_col_isdf(const double* __restrict__ A,
                                      int* Imin, double* amin,
                                      int m, int n, bool computeMin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   
    if (i >= m) return;

    double minValue = A[i];                          
    int    minIndex = 0;
    for (int j = 1; j < n; j++) {
        double v = A[(size_t)j * m + i];
        if (v < minValue) { minValue = v; minIndex = j; }
    }
    Imin[i] = minIndex;
    if (computeMin) amin[i] = minValue;
}



__global__ void findminIndex_row_isdf(const double* __restrict__ A,
                                      int* Imin, double* amin,
                                      int m, int n, bool computeMin)
{
    int col = blockIdx.x;
    if (col >= n) return;

    constexpr int BS = 256;
    __shared__ double sval[BS];
    __shared__ int    sidx[BS];

    int tid = threadIdx.x;
    double myVal = DBL_MAX;
    int    myIdx = 0;
        for (int j = tid; j < m; j += BS) {
        double v = A[(size_t)col * m + j];
        if (v < myVal) { myVal = v; myIdx = j; }
    }
    sval[tid] = myVal;
    sidx[tid] = myIdx;
    __syncthreads();

    #pragma unroll
    for (int s = BS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double vo = sval[tid + s];
            double vt = sval[tid];
            int    io = sidx[tid + s];
            int    it = sidx[tid];
            if (vo < vt || (vo == vt && io < it)) {
                sval[tid] = vo;
                sidx[tid] = io;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        Imin[col] = sidx[0];
        if (computeMin) amin[col] = sval[0];
    }
}
















__global__ void computeCuvT_isdf(double* psiCol, int * pivQR, double* phiMuv,
                                 int rk, int ntot, int num_local, int idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rk && j < num_local) {
        phiMuv[i * num_local + j] = psiCol[pivQR[i] + ntot * (j + idx)];
    }
}

// gpuAssert helper (file-local — static inline avoids ODR issues)
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


// ============================================================
// All host-side globals and functions go inside namespace isdf
// ============================================================
namespace isdf {

double *dev_vtot;
double *dev_gkkR2C;
int    *dev_idxFineGridR2C;
int    *dev_NLindex;
int    *dev_NLpart;
double *dev_NLvecFine;
double *dev_atom_weight;
double *dev_temp_weight;
cuDoubleComplex *dev_temp_weight_complex;
double *dev_TeterPrecond;

bool vtot_gpu_flag;
bool NL_gpu_flag;
bool teter_gpu_flag;
int  totPart_gpu;


void *cuda_malloc(size_t size)
{
    void *ptr;
    CUDA_CALL(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_memory(void)
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    cudaMemGetInfo(&free_mem, &total_mem);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    printf("free  memory is: %zu MB\n", free_mem / 1000000);
    printf("total memory is: %zu MB\n", total_mem / 1000000);
    fflush(stdout);
}

void cuda_sync()
{
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    assert(cudaDeviceSynchronize() == cudaSuccess);
}

void cuda_memcpy_CPU2GPU(void *gpu, void *cpu, size_t size)
{
    CUDA_CALL(cudaMemcpy(gpu, cpu, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_GPU2CPU(void *cpu, void *gpu, size_t size)
{
    CUDA_CALL(cudaMemcpy(cpu, gpu, size, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_GPU2GPU(void *dest, void *src, size_t size)
{
    CUDA_CALL(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
}

void cuda_setValue(double *dev, double val, int len)
{
    int ndim = len / DIM;
    if (len % DIM) ndim++;
    gpu_setValue_isdf<<<ndim, DIM>>>(dev, val, len);
#ifdef SYNC
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

void cuda_free(void *ptr)
{
    CUDA_CALL(cudaFree(ptr));
}

void getelemproduct(int rk1, int rk2, double* C, const double* A, const double* B)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((rk1 + blockSize.x - 1) / blockSize.x,
                  (rk2 + blockSize.y - 1) / blockSize.y);
    elemproduct_isdf<<<gridSize, blockSize>>>(rk1, rk2, C, A, B);
}

void calculateKmeansWeight(const double* d_psi, int rows,
                           int i0_1, int i1_1,
                           int i0_2, int i1_2,
                           double* d_weightLocal,
                           cudaStream_t stream)
{
    constexpr int BS = 256;
    dim3 grid(rows);
    dim3 block(BS);
    kmeansWeightKernel_isdf<BS><<<grid, block, 0, stream>>>(
        d_psi, rows, i0_1, i1_1, i0_2, i1_2, d_weightLocal);
}
/*
void pdist2_GPU(double* A, double* B, double* DD, int na, int nb)
{
    dim3 blockDim_(16, 16);
    dim3 gridDim_((na + blockDim_.x - 1) / blockDim_.x,
                  (nb + blockDim_.y - 1) / blockDim_.y);
    computeDistances_isdf<<<gridDim_, blockDim_>>>(A, B, DD, na, nb);
}
*/


void pdist2_GPU(double* A, double* B, double* DD, int na, int nb)
{
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 16;
    static_assert(TILE_M * TILE_N <= 1024, "block size exceeds limit");
    
    dim3 block(TILE_M, TILE_N);
    dim3 grid((na + TILE_M - 1) / TILE_M,
              (nb + TILE_N - 1) / TILE_N);
    computeDistances_isdf_tiled<TILE_M, TILE_N><<<grid, block>>>(A, B, DD, na, nb);
}


/*
void findMin_GPU(double* A, int* Imin, double* amin, int col, int m, int n, bool computeMin)
{
    dim3 blockDim_(16, 16);
    dim3 gridDim_((m + blockDim_.x - 1) / blockDim_.x,
                  (n + blockDim_.y - 1) / blockDim_.y);
    if (col == 1) {
        findminIndex_col_isdf<<<gridDim_, blockDim_>>>(A, Imin, amin, m, n, computeMin);
    } else {
        findminIndex_row_isdf<<<gridDim_, blockDim_>>>(A, Imin, amin, m, n, computeMin);
    }
}
*/


void findMin_GPU(double* A, int* Imin, double* amin,
                 int col, int m, int n, bool computeMin)
{
    constexpr int BS = 256;
    if (col == 1) {
            if (m > 0) {
             int grid = (m + BS - 1) / BS;  
             findminIndex_col_isdf<<<grid, BS>>>(A, Imin, amin, m, n, computeMin);
 //           findminIndex_col_isdf<<<m, BS>>>(A, Imin, amin, m, n, computeMin);
        }
    } else {
        if (n > 0) {
            findminIndex_row_isdf<<<n, BS>>>(A, Imin, amin, m, n, computeMin);
        }
    }
}

void cuda_gather_rows(const double* d_src, const int* d_piv,
                      int rk, int ncol, int ldSrc, double* d_dst)
{
    dim3 block(32, 8);
    dim3 grid((ncol + block.x - 1) / block.x,
              (rk   + block.y - 1) / block.y);
    gather_rows_kernel<<<grid, block>>>(d_src, d_piv, rk, ncol, ldSrc, d_dst);
}





void cuda_scatter_cols(const double* d_src, const int* d_ownedNu,
                       int rk, int selectedRow, double* d_dst)
{


if (selectedRow <= 0) return; 
    dim3 block(32, 8);
    dim3 grid((selectedRow + block.x - 1) / block.x,
              (rk          + block.y - 1) / block.y);
    scatter_cols_kernel<<<grid, block>>>(d_src, d_ownedNu, rk, selectedRow, d_dst);
}

void cuda_add_diag(double* d_A, int n, double eps)
{
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    add_diag_kernel<<<grid, block>>>(d_A, n, eps);
}



void getpsiMuT(double *psiCol, int *pivQR, double* phiMuv,
               int rk, int ntot, int num_local, int idx)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((rk + blockSize.x - 1) / blockSize.x,
                  (num_local + blockSize.y - 1) / blockSize.y);
    computeCuvT_isdf<<<gridSize, blockSize>>>(psiCol, pivQR, phiMuv, rk, ntot, num_local, idx);
}

} // namespace isdf
