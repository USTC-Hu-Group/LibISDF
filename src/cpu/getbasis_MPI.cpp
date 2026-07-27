#include "../../include/isdf.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
namespace isdf
{

#if !defined(GPU) 
  void isdffunc::getbasis_MPI(DblNumMat &psiRow, DblNumMat &psiCol, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, DblNumMat &psiphizetaRow, IntNumVec &pivQR_, Real &timeISDFMPI)

  {

    Real timeSta, timeStaisdf, timeStaMPI, timeStaGemm, timeSta1;
    Real timeEnd, timeEndisdf, timeEndMPI, timeEndGemm, timeEnd1;
    Real timeGemm = 0.0;
    Real timeMPI = 0.0;
    GetTime(timeStaisdf);
    MPI_Barrier(domain_.comm);
    int mpirank;
    MPI_Comm_rank(domain_.comm, &mpirank);
    int mpisize;
    MPI_Comm_size(domain_.comm, &mpisize);
    Int ntot_ = domain_.NumGridTotal();
    //GetTime(timeSta);


    DblNumMat PMuNu;
    Int ntotBlocksize = ntot_ / mpisize;
    IntNumVec ntotresTotal(mpisize);
    SetValue(ntotresTotal, 0);

    if ((ntot_ % mpisize) == 0)
    {
      for (Int p = 0; p < mpisize; p++)
      {
        ntotresTotal(p) = p * ntotBlocksize;
      }
    }
    else
    {
      for (Int p = 0; p < mpisize; p++)
      {
        if (p < (ntot_ % mpisize))
        {
          ntotresTotal(p) = p * (ntotBlocksize + 1);
        }
        else
        {
          ntotresTotal(p) =
              (ntot_ % mpisize) * (ntotBlocksize + 1) + (p - (ntot_ % mpisize)) * ntotBlocksize;
        }
      }
    }

    Int rkLocal = rk / mpisize;
    if (mpirank < (rk % mpisize))
    {
      rkLocal++;
    }
    DblNumMat psiMuLocalCol(rk, psiCol.n());
    SetValue(psiMuLocalCol, 0.0);
    for (int i = 0; i < psiCol.n(); i++)
    {
      for (int j = 0; j < rk; j++)
      {
        psiMuLocalCol(j, i) = psiCol(pivQR_[j], i);
      }
    }
    DblNumMat psiMuLocalRow(rkLocal, psiRow.n());
    SetValue(psiMuLocalRow, 0.0);
    GetTime(timeStaMPI);
    AlltoallForward(psiMuLocalCol, psiMuLocalRow, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI+=timeEndMPI-timeStaMPI;
    DblNumMat psiMuLocalRowT(psiRow.n(), rkLocal);
    SetValue(psiMuLocalRowT, 0.0);

    for (int i = 0; i < psiRow.n(); i++)
    {
      for (int j = 0; j < rkLocal; j++)
      {
        psiMuLocalRowT(i, j) = psiMuLocalRow(j, i);
      }
    }

    IntNumVec widthLocals(mpisize);
    GetTime(timeStaMPI);
    MPI_Allgather(&rkLocal, 1, MPI_INT, widthLocals.Data(), 1, MPI_INT, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI+=timeEndMPI-timeStaMPI;
    DblNumMat psiMuRow(psiRow.n(), rk);
    SetValue(psiMuRow, 0.0);

    IntNumVec sendcountsum(mpisize);
    SetValue(sendcountsum, 0);

    IntNumVec displspsi(mpisize);
    SetValue(displspsi, 0);

    for (int i = 0; i < mpisize; i++)
    {
      sendcountsum[i] = widthLocals[i] * psiRow.n();
    }

    for (int i = 1; i < mpisize; i++)
    {
      displspsi[i] = displspsi[i - 1] + widthLocals[i - 1] * psiRow.n();
    }
    GetTime(timeStaMPI);
    MPI_Allgatherv(psiMuLocalRowT.Data(), rkLocal * psiRow.n(), MPI_DOUBLE, psiMuRow.Data(), sendcountsum.Data(), displspsi.Data(), MPI_DOUBLE, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI+=timeEndMPI-timeStaMPI;

    psiMuLocalCol.FreeData();
    psiMuLocalRow.FreeData();
    psiMuLocalRowT.FreeData();

    DblNumMat PpsimuLocal(ntotLocal, rk);
    SetValue(PpsimuLocal, 0.0);
    GetTime(timeSta);

    if (nv1 != nv2 || nc1 != nc2)
    {

      GetTime(timeStaGemm);
      blas::Gemm('N', 'N', ntotLocal, rk, nv1 + nc1, 1.0, psiRow.Data() + (nv_ - nv1) * ntotLocal, ntotLocal, psiMuRow.Data() + (nv_ - nv1), psiRow.n(), 0.0, PpsimuLocal.Data(), ntotLocal);
      GetTime(timeEndGemm);
      timeGemm = timeGemm + timeEndGemm - timeStaGemm;
      isdfOFS << "rk=" << rk << std::endl;
      DblNumMat PphimuLocal(ntotLocal, rk);
      SetValue(PphimuLocal, 0.0);
      GetTime(timeStaGemm);
      blas::Gemm('N', 'N', ntotLocal, rk, nv2 + nc2, 1.0, psiRow.Data() + (nv_ - nv2) * ntotLocal, ntotLocal, psiMuRow.Data() + (nv_ - nv2), psiRow.n(), 0.0, PphimuLocal.Data(), ntotLocal);
      GetTime(timeEndGemm);
      timeGemm = timeGemm + timeEndGemm - timeStaGemm;
      GetTime(timeEnd);
      isdfOFS << "Time for Pphimu and Pphimu       = " << timeEnd - timeSta << " [s]" << std::endl;

      GetTime(timeSta);

      for (int i = 0; i < rk; i++)
      {
        for (int j = 0; j < ntotLocal; j++)
        {
          psiphizetaRow(j, i) = PpsimuLocal(j, i) * PphimuLocal(j, i);
        }
      }

      GetTime(timeEnd);
      PphimuLocal.FreeData();
      PpsimuLocal.FreeData();
      isdfOFS << "Time for Z*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;
    }
    else
    {
//      GetTime(timeEnd);
//      isdfOFS << "Time for phiMu and psiMu         = " << timeEnd - timeSta << " [s]" << std::endl;
      GetTime(timeStaGemm);
      blas::Gemm('N', 'N', ntotLocal, rk, nv1 + nc1, 1.0, psiRow.Data() + (nv_ - nv1) * ntotLocal, ntotLocal, psiMuRow.Data() + (nv_ - nv1), psiRow.n(), 0.0, PpsimuLocal.Data(), ntotLocal);
      GetTime(timeEndGemm);
      timeGemm = timeGemm + timeEndGemm - timeStaGemm;
      GetTime(timeEnd);
      isdfOFS << "Time for Pphimu and Pphimu       = " << timeEnd - timeSta << " [s]" << std::endl;

      GetTime(timeSta);

      for (int i = 0; i < rk; i++)
      {
        for (int j = 0; j < ntotLocal; j++)
        {
          psiphizetaRow(j, i) = PpsimuLocal(j, i) * PpsimuLocal(j, i);
        }
      }

      GetTime(timeEnd);
      PpsimuLocal.FreeData();
      isdfOFS << "Time for Z*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;
    }
    psiMuRow.FreeData();

    GetTime(timeSta);
    PMuNu.Resize(rk, rk);
    SetValue(PMuNu, 0.0);

    for (Int nu = 0; nu < rk; nu++)
    {
      Int globalIdx = pivQR_(nu);
      bool owned = false;
      if (mpirank < mpisize - 1)
      {
        owned = globalIdx >= ntotresTotal(mpirank) &&
                globalIdx < ntotresTotal(mpirank + 1);
      }
      else
      {
        owned = globalIdx >= ntotresTotal(mpirank) &&
                globalIdx < ntot_;
      }
      if (owned)
      {
        Int localIdx = globalIdx - ntotresTotal(mpirank);

        for (Int mu = 0; mu < rk; mu++)
        {
          PMuNu(mu, nu) = psiphizetaRow(localIdx, mu);
        }
      }
    }

    GetTime(timeStaMPI);
    MPI_Allreduce(MPI_IN_PLACE, PMuNu.Data(), rk * rk, MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI += timeEndMPI - timeStaMPI;
    GetTime(timeEnd);
    isdfOFS << "Time for C*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;
    double traceP = 0.0;
    for (int i = 0; i < rk; i++)
    {
      traceP += PMuNu(i, i);
    }
    double eps = 1e-10 * (traceP / double(rk));
    for (int i = 0; i < rk; i++)
    {
      PMuNu(i, i) += eps;
    }

    GetTime(timeSta);
    lapack::Potrf('L', rk, PMuNu.Data(), rk);
    GetTime(timeEnd);
    isdfOFS << "Time for Potrf                   = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
    blas::Trsm('R', 'L', 'T', 'N', ntotLocal, rk, 1.0, PMuNu.Data(), rk, psiphizetaRow.Data(), ntotLocal);
    blas::Trsm('R', 'L', 'N', 'N', ntotLocal, rk, 1.0, PMuNu.Data(), rk, psiphizetaRow.Data(), ntotLocal);
    GetTime(timeEnd);
    isdfOFS << "Time for Trsm                    = " << timeEnd - timeSta << " [s]" << std::endl;
    GetTime(timeEndisdf);
    isdfOFS << "Time for getbais       = " << timeEndisdf - timeStaisdf << " [s]" << std::endl;
    isdfOFS << "ISDF Time for GEMM in  get basis       = " << timeGemm << " [s]" << std::endl;
    isdfOFS << "ISDF Time for MPI in   get basis      = " << timeMPI << " [s]" << std::endl;
    timeISDFMPI = timeISDFMPI + timeMPI;
//    timeISDFGEMM = timeISDFGEMM + timeGemm;
  }

#endif
}
