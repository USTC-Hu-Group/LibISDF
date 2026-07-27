#include "../../include/isdf.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
namespace isdf{

#if !defined(GPU) 
void isdffunc::ISDF(double *psirow, Domain domain_, Int nocc, Int Nstate, Int nv1, Int nc1, Int nv2, Int nc2, Int mu_rank, Real *thetaRow, Int *piv, Int KmeansMaxIter_ISDF, Int scalblocksize,std::string s , std::string s1,bool check)
{
  int mpirank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Barrier(domain_.comm);
  Int nr = domain_.NumGridTotal();
  Int ntotLocal = nr / mpisize;
  Real timeStaisdf;
  Real timeEndisdf;
  Real timeISDF = 0.0;
  Real timeISDFMPI = 0.0;
  Real timeSta,timeEnd;

  GetTime(timeStaisdf);
  if (mpirank < (nr % mpisize))
  {
   ntotLocal++;
  }

  DblNumMat psiRow(ntotLocal, Nstate, false, psirow);
  Int NstateLocal = Nstate / mpisize;
  if (mpirank < (Nstate % mpisize))
  {
    NstateLocal++;
  }

  
  Int rk;
  if(mu_rank == 0){
  rk = IRound(std::sqrt((nc1 + nv1) * (nc2 + nv2)) * 8.0);
  }else{
	  rk = mu_rank;
  }
  isdfOFS<<"rk="<< rk <<std::endl;
  IntNumVec pivQR_(nr, false, piv);
  SetValue(pivQR_, 0);
  Int rkLocal = rk / mpisize;
  if (mpirank < (rk % mpisize))
  {
    rkLocal++;
  }

  DblNumMat psiphirzetaCol(nr, rkLocal, false, thetaRow);
  DblNumMat psiphirzetaRow(ntotLocal, rk);
  SetValue(psiphirzetaRow, 0.0);
  //printCpxM(psiRow, isdfOFS);
  DblNumMat psiCol(nr, NstateLocal);
  GetTime(timeSta);
  AlltoallBackward(psiRow, psiCol, domain_.comm);
  GetTime(timeEnd);
  timeISDFMPI+=timeEnd-timeSta;
  if (s == "MPI")
  {
    if (s1 == "Kmeans")
    {
      getpoints_Kmeans_MPI(psiRow, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, pivQR_, timeISDFMPI,KmeansMaxIter_ISDF);
    }
    else if(s1== "QRCP")
    {
      srand48(41);
      getpoints_QRCP(psiCol, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, pivQR_);
    }
    else
  {
    ErrorHandling("Wrong select points method");
  }
   // printCpxV(pivQR_,psiOFS);
    getbasis_MPI(psiRow,psiCol, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, psiphirzetaRow, pivQR_, timeISDFMPI);
    GetTime(timeSta);
    AlltoallBackward(psiphirzetaRow, psiphirzetaCol, domain_.comm);
    GetTime(timeEnd);
    timeISDFMPI+=timeEnd-timeSta;
    GetTime(timeEndisdf);
    timeISDF = timeEndisdf-timeStaisdf;
    //printCpxM(psiphirzetaCol,psiOFS);

    
    
    isdfOFS << "Time for ISDF ="<<timeEndisdf-timeStaisdf  << " [s]" <<std::endl;
    if (check)
    {
      isdf::Fourier fft;
      fft.Initialize(domain_);
      checkcode(psiRow, psiCol, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, psiphirzetaRow, pivQR_,fft);
    }
  }
  else if (s == "scalapack")
  {

    if (s1 == "Kmeans")
    {
      getpoints_Kmeans_MPI(psiRow, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, pivQR_, timeISDFMPI,KmeansMaxIter_ISDF);
    }
    else if(s1== "QRCP")
    {
      srand48(41);
      getpoints_QRCP(psiCol, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, pivQR_);
    }
    else
  {
    ErrorHandling("Wrong select points method");
  }


    getbasis_scalapack(psiCol, domain_, Nstate, nocc, nv1, nc1, nv2, nc2, rk, psiphirzetaCol, pivQR_,scalblocksize, timeISDFMPI);
    GetTime(timeEndisdf);
    isdfOFS << "Time for ISDF ="<<timeEndisdf-timeStaisdf  << " [s]" <<std::endl;
    if (check)
    {
    AlltoallForward(psiphirzetaCol, psiphirzetaRow, domain_.comm);
    isdf::Fourier fft;
    fft.Initialize(domain_);
    isdffunc::checkcode(psiRow, psiCol, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, psiphirzetaRow, pivQR_,fft);
    }
  }
}


#else

void isdffunc::ISDF(double *d_psirow, Domain domain_, Int nocc, Int Nstate, Int nv1, Int nc1, Int nv2, Int nc2, Int mu_rank, Real *d_thetaCol, Int *piv,Int KmeansMaxIter_ISDF,bool check)
{
  int mpirank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Barrier(domain_.comm);
  Int nr = domain_.NumGridTotal();
  Int ntotLocal = nr / mpisize;
  if (mpirank < (nr % mpisize))
  {
    ntotLocal++;
  }

  cuDblNumMat cu_psiRow(ntotLocal, Nstate, false, d_psirow);

  Int NstateLocal = Nstate / mpisize;
  if (mpirank < (Nstate % mpisize))
  {
    NstateLocal++;
  }

  Real timeISDF = 0.0;
  Real timeISDFMPI = 0.0;
  Real timeSta, timeEnd;
  Real timeSta1,timeEnd1;
//  Real timeISDFGEMM = 0.0;
  Real timeISDFMemcpy=0.0;
  Int rk;
  if(mu_rank == 0){
  rk = IRound(std::sqrt((nc1 + nv1) * (nc2 + nv2)) * 8.0);
  }else{
          rk = mu_rank;
  }
  isdfOFS<<"rk="<< rk <<std::endl;
  Int rkLocal = rk / mpisize;
  if (mpirank < (rk % mpisize))
  {
    rkLocal++;
  }


  cuDblNumMat cu_psiphirzetaCol(nr, rkLocal, false, d_thetaCol);
  IntNumVec pivQR_(nr, false, piv);
  SetValue(pivQR_, 0);
  cuDblNumMat cu_psiphirzetaRow(ntotLocal, rk);
  Real timeStaisdf;
  Real timeEndisdf;
  
  GetTime(timeStaisdf);
  ISDF_GPU(cu_psiRow, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, cu_psiphirzetaRow, pivQR_,KmeansMaxIter_ISDF,timeISDFMemcpy,timeISDFMPI);
  GetTime(timeSta);
  GPU_AlltoallBackward(cu_psiphirzetaRow,cu_psiphirzetaCol,domain_.comm); 
  GetTime(timeEnd);
  timeISDFMPI+=timeEnd-timeSta;
  MPI_Barrier(domain_.comm);
  GetTime(timeEndisdf);
   
  isdfOFS << "Time for ISDF time =" << timeEndisdf - timeStaisdf << " [s]" << std::endl;
  isdfOFS << "Time for CPUGPUmemcpy =" << timeISDFMemcpy << " [s]" << std::endl;
  isdfOFS << "Time for MPI =" << timeISDFMPI << " [s]" << std::endl;

  if (check)
  {
    DblNumMat psiphirzetaRow(ntotLocal, rk);
    GetTime(timeSta1);
    cuda_memcpy_GPU2CPU(psiphirzetaRow.Data(), cu_psiphirzetaRow.Data(), rk * ntotLocal * sizeof(double));
    GetTime(timeEnd1);
    isdf::Fourier fft;
    fft.Initialize(domain_);
    DblNumMat psiRow(ntotLocal, Nstate);
    cuda_memcpy_GPU2CPU(psiRow.Data(), cu_psiRow.Data(), ntotLocal * Nstate * sizeof(double));
    DblNumMat psiCol(nr, NstateLocal);
    AlltoallBackward(psiRow, psiCol, domain_.comm);
    isdffunc::checkcode(psiRow, psiCol, domain_, nocc, nv1, nc1, nv2, nc2, ntotLocal, rk, psiphirzetaRow, pivQR_,fft);
  }

 // isdfOFS.close();
}

#endif
}
