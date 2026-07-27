#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <unistd.h>
#include "isdf.hpp"
#if defined(GPU) 
#include <cuda_runtime.h>
#endif
using namespace isdf;
class MatrixReader
{
private:
    int rows;
    int cols;
    double **matrix;
    double *data_block;

public:
    MatrixReader() : rows(0), cols(0), matrix(nullptr), data_block(nullptr) {}

    ~MatrixReader()
    {
        cleanup();
    }
    MatrixReader(const MatrixReader&)            = delete;
    MatrixReader& operator=(const MatrixReader&) = delete;

    bool readFromFile(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        file >> rows >> cols;
        if (rows <= 0 || cols <= 0)
            return false;
        matrix = new double *[cols]; 
        data_block = new double[rows * cols];
        for (int i = 0; i < cols; ++i)
        {
            matrix[i] = data_block + i * rows; 
        }
       
        for (int i = 0; i < cols; i++)
        { 
            for (int j = 0; j < rows; j++)
            { 
                file >> matrix[i][j];
            }
        }

        return true;
    }


    double **getMatrix() const { return matrix; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }

private:
    void cleanup()
    {
        if (matrix)
        {
            delete[] matrix;
            matrix = nullptr;
        }
        if (data_block)
        {
            delete[] data_block;
            data_block = nullptr;
        }
    }
};

int main(int argc, char *argv[])
{


fprintf(stderr, "PID=%d ENTERING main\n", getpid());
fflush(stderr);

MPI_Init(&argc, &argv);

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
fprintf(stderr, "PID=%d AFTER MPI_Init: rank=%d size=%d\n", getpid(), rank, size);
fflush(stderr);

#if defined(GPU)





    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int ngpus = 0;
    cudaError_t err = cudaGetDeviceCount(&ngpus);
    if (err != cudaSuccess || ngpus == 0) {
        fprintf(stderr, "rank=%d cudaGetDeviceCount failed: %s (ngpus=%d)\n",
                rank, cudaGetErrorString(err), ngpus);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev = local_rank % ngpus;
    cudaSetDevice(dev);


    cudaFree(0);

    int cur = -1;
    cudaGetDevice(&cur);
    fprintf(stderr, "[GPU bind] rank=%d local_rank=%d ngpus=%d -> device=%d\n",
            rank, local_rank, ngpus, cur);
    fflush(stderr);

#endif


    Domain domain_;

    domain_.length = Point3(10.216, 10.216, 10.216);
    domain_.posStart = Point3(0.0, 0.0, 0.0);
    domain_.numGrid = Index3(16, 16, 16);
    domain_.numGridFine = Index3(32, 32, 32);

    domain_.comm = MPI_COMM_WORLD;
    domain_.rowComm = MPI_COMM_WORLD;
    domain_.colComm = MPI_COMM_WORLD;
    MatrixReader readerrow;
    double **psirow = nullptr;
    int rows = 0, cols = 0;
    std::string filenamerow = "examples/si8/data/psirow" + std::to_string(rank);



    if (!readerrow.readFromFile(filenamerow)) {
    std::cerr << "rank " << rank << " failed to open " << filenamerow << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;}

        psirow = readerrow.getMatrix();
        rows = readerrow.getRows();
        cols = readerrow.getCols();

        //Int mu_rank = 128;
	Int mu_num=192;

        Int Nr = domain_.NumGridTotal();
        Int nstate = 35;
        Int Nc1 = 0;
        Int Nc2 = 0;
        Int Nv1 = 16;
        Int Nv2 = 16;
        Int nocc = 16;
//	if(rank==0)
//	{
//        std::stringstream ss1;
//        ss1 << "psiisdf." << rank;
//        psiOFS.open(ss1.str().c_str());
//	}
  	if (rank == 0)
  	{
 	 std::stringstream ss;
  	 ss << "isdf." <<rank;
 	 isdfOFS.open(ss.str().c_str());
 	 }
	Int Nu = mu_num;

  //      Int Nu = IRound(std::sqrt((Nc1 + Nv1) * (Nc2 + Nv2)) * mu_rank);
        Int NrLocal = Nr / size;
        if (rank < Nr % size)
        {
            NrLocal++;
        }


        Int NuLocal = Nu / size;
        if (rank < Nu % size)
        {   
            NuLocal++;
        }
        double *psiphizetaData = new double[NuLocal * Nr];
        double **psiphizetaCol = new double *[NuLocal];
        for (int i = 0; i < NuLocal; i++)
        {
            psiphizetaCol[i] = &psiphizetaData[i * Nr];
        }

#if defined(GPU)
        double *d_psiphizetaData;
        size_t gpu_size = NuLocal * Nr * sizeof(double);
        cudaMalloc((void **)&d_psiphizetaData, gpu_size);

        cudaMemcpy(d_psiphizetaData, psiphizetaData, gpu_size, cudaMemcpyHostToDevice);
#endif


        Int *piv = new Int[Nr]; 

       
        std::cout << "Process " << rank << ": Calling ISDF function..." << std::endl;

        double Sta;
        double End;

        GetTime(Sta);
        MPI_Barrier(MPI_COMM_WORLD);
        bool check=true;
        int KmeansMaxIter_ISDF=500;
#if !defined(GPU)
        std::string s = "scalapack";
        //std::string s="MPI";
        std::string s1 = "Kmeans";
        //std::string s1="QRCP";
        std::cout<<s<<std::endl;
        std::cout<<s1<<std::endl;
        int blocksize=32;
	isdffunc::ISDF(psirow[0], domain_, nocc, nstate, Nv1, Nc1, Nv2, Nc2, Nu, psiphizetaCol[0], piv,KmeansMaxIter_ISDF,blocksize, s, s1,check);
#endif
#if defined(GPU)

        double *d_psirow;
        size_t psi_size = rows * cols * sizeof(double);
        cudaMalloc((void **)&d_psirow, psi_size);
        cudaMemcpy(d_psirow, psirow[0], psi_size, cudaMemcpyHostToDevice);
	isdffunc::ISDF(d_psirow, domain_, nocc, nstate, Nv1, Nc1, Nv2, Nc2, Nu, d_psiphizetaData, piv,KmeansMaxIter_ISDF,check);
        
        cudaMemcpy(psiphizetaData, d_psiphizetaData, gpu_size, cudaMemcpyDeviceToHost);
#endif
        MPI_Barrier(MPI_COMM_WORLD);
        GetTime(End);
        
        std::cout << "Time " << rank<<"  " << End - Sta << std::endl;
        GetTime(Sta);
        std::cout << "Process " << rank << ": GW_ISDF function completed." << std::endl;
        std::cout << "Process " << rank << ": Freeing memory..." << std::endl;

        delete[] piv;
        std::cout << "Process " << rank << ": piv freed." << std::endl;

#if defined(GPU)
        cudaFree(d_psiphizetaData);
        cudaFree(d_psirow);
#endif

        delete[] psiphizetaCol;
        std::cout << "Process " << rank << ": psiphizetaCol freed." << std::endl;

        delete[] psiphizetaData;
        std::cout << "Process " << rank << ": psiphizetaData freed." << std::endl;

        std::cout << "Process " << rank << ": All memory freed successfully." << std::endl;
        GetTime(End);
        std::cout << "Time " << rank<<"  " << End - Sta << std::endl;


    isdfOFS.close();
    psiOFS.close();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
