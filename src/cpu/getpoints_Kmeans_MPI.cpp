#include "../../include/isdf.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
namespace isdf{
#if !defined(GPU)

void isdffunc::getpoints_Kmeans_MPI(DblNumMat &psiRow, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, IntNumVec &pivQR_, Real &timeISDFMPI,Int &KmeansMaxIter_ISDF)
{
  Real timeSta, timeStaMPI;
  Real timeEnd, timeEndMPI;
  Real timeMPI = 0.0;
  GetTime(timeSta);
  MPI_Barrier(domain_.comm);
  int mpirank;
  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;
  MPI_Comm_size(domain_.comm, &mpisize);
  Real KmeansTolerance = 1e-5;
  Real Tolerance = 1e-8;
  Int ntot_ = domain_.NumGridTotal();
  {
    IntNumVec weightSizeDispls(mpisize);
    IntNumVec weightSize(mpisize);
    DblNumVec weightLocal(ntotLocal);
    SetValue(weightLocal, 0.0);
    DblNumVec weight(ntot_);
    SetValue(weight, 0.0);
    Int ntotBlocksize = ntot_ / mpisize;
    DblNumVec sumI(ntotLocal);
    SetValue(sumI, 0.0);
    DblNumVec sumJ(ntotLocal);
    SetValue(sumJ, 0.0);
    for (int i = nv_ - nv2; i < nv_ + nc2; i++)
      for (int kk = 0; kk < ntotLocal; kk++)
        sumI(kk) += psiRow(kk, i) * psiRow(kk, i);
    for (int j = nv_ - nv1; j < nv_ + nc1; j++)
       for (int kk = 0; kk < ntotLocal; kk++)
        sumJ(kk) += psiRow(kk, j) * psiRow(kk, j);

    for (int kk = 0; kk < ntotLocal; kk++)
       weightLocal(kk) += sumI(kk) * sumJ(kk);

    sumI.FreeData();
    sumJ.FreeData();

    if ((ntot_ % mpisize) == 0)
    {
      for (Int i = 0; i < mpisize; i++)
      {
        weightSizeDispls[i] = i * ntotBlocksize;
        weightSize[i] = ntotBlocksize;
      }
    }
    else
    {
      for (Int i = 0; i < mpisize; i++)
      {
        if (i < (ntot_ % mpisize))
        {
          weightSizeDispls[i] = i * (ntotBlocksize + 1);
          weightSize[i] = ntotBlocksize + 1;
        }
        else
        {
          weightSizeDispls[i] = (ntot_ % mpisize) * (ntotBlocksize + 1) + (i - (ntot_ % mpisize)) * (ntotBlocksize);
          weightSize[i] = ntotBlocksize;
        }
      }
    }
    GetTime(timeStaMPI);
    MPI_Allgatherv(weightLocal.Data(), ntotLocal, MPI_DOUBLE, weight.Data(), weightSize.Data(), weightSizeDispls.Data(), MPI_DOUBLE, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI = timeMPI + timeEndMPI - timeStaMPI;
    timeISDFMPI=timeISDFMPI+timeMPI;
    KMEAN(ntot_, weight, rk, KmeansTolerance, KmeansMaxIter_ISDF, Tolerance, domain_, pivQR_.Data());
    GetTime(timeEnd);
    isdfOFS << "Time for pivQR_ with Kmeans      = " << timeEnd - timeSta << " [s]" << std::endl;
    isdfOFS << "Time for MPI in  Kmeans      = " << timeMPI << " [s]" << std::endl;
  }
}



#endif
}
