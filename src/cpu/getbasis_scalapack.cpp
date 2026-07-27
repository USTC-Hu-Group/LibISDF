#include "../../include/isdf.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
using namespace isdf::scalapack;
namespace isdf
{
#if !defined(GPU)
  void isdffunc::getbasis_scalapack(DblNumMat &psiCol, Domain &domain_, const Int &Nstate, const Int &nocc, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int Nu, DblNumMat &Xi, IntNumVec &pivQR_, Int &scalblocksize,Real &timeISDFMPI)
  {
    MPI_Barrier(domain_.comm);
    int mpirank;
    MPI_Comm_rank(domain_.comm, &mpirank);
    Int mpisize;
    MPI_Comm_size(domain_.comm, &mpisize);
//    Real timeISDFGEMM=0.0;
    Real timeSta, timeStaisdf, timeStaMPI, timeStaGemm;
    Real timeEnd, timeEndisdf, timeEndMPI, timeEndGemm;
    Real timeGemm = 0.0;
    Real timeMPI = 0.0;
    GetTime(timeStaisdf);
    Int Ng = domain_.NumGridTotal();
    Int Nb = Nstate;
    Int I_ONE = 1, I_ZERO = 0;
    Real D_ONE = 1.0, D_ZERO = 0.0;
    Int Basis_BlockSize = 1;
    Int Grid_BlockSize = scalblocksize;
    Real timeSta1, timeEnd1;
    Real timeSta2, timeEnd2;
    Int contxt1DRow, contxt1DCol, contxt2D;
    Int nprow1DRow, npcol1DRow, myrow1DRow, mycol1DRow, info1DRow;
    Int nprow1DCol, npcol1DCol, myrow1DCol, mycol1DCol, info1DCol;
    Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

    Int nrowsNgNu1DRow, ncolsNgNu1DRow, lldNgNu1DRow;
    Int nrowsNgNb1DRow, ncolsNgNb1DRow, lldNgNb1DRow;
    Int nrowsNbNu1DRow, ncolsNbNu1DRow, lldNbNu1DRow;
    Int nrowsNgNu1DCol, ncolsNgNu1DCol, lldNgNu1DCol;
    Int nrowsNgNb1DCol, ncolsNgNb1DCol, lldNgNb1DCol;
    Int nrowsNuNu1DCol, ncolsNuNu1DCol, lldNuNu1DCol;

    Int nrowsNgNu2D, ncolsNgNu2D, lldNgNu2D;
    Int nrowsNgNb2D, ncolsNgNb2D, lldNgNb2D;
    Int nrowsNuNu2D, ncolsNuNu2D, lldNuNu2D;
    Int nrowsNbNu2D, ncolsNbNu2D, lldNbNu2D;

    Int desc_NgNb1DRow[9];
    Int desc_NgNu1DRow[9];
    Int desc_NbNu1DRow[9];
    Int desc_NuNb1DRow[9];

    Int desc_NgNb1DCol[9];
    Int desc_NgNu1DCol[9];
    Int desc_NuNu1DCol[9];

    Int desc_NgNu2D[9];
    Int desc_NuNu2D[9];
    Int desc_NgNb2D[9];
    Int desc_NbNu2D[9];

    IntNumVec isdf_pmap(mpisize);
    for (Int pmap_iter = 0; pmap_iter < mpisize; pmap_iter++)
    {
      isdf_pmap[pmap_iter] = pmap_iter;
    }

    nprow1DCol = 1;
    npcol1DCol = mpisize;

    Cblacs_get(0, 0, &contxt1DCol);
    Cblacs_gridmap(&contxt1DCol, &isdf_pmap[0], nprow1DCol, nprow1DCol, npcol1DCol);
    Cblacs_gridinfo(contxt1DCol, &nprow1DCol, &npcol1DCol, &myrow1DCol, &mycol1DCol);

    if (contxt1DCol >= 0)
    {
      nrowsNgNu1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
      ncolsNgNu1DCol = SCALAPACK(numroc)(&Nu, &Basis_BlockSize, &mycol1DCol, &I_ZERO, &npcol1DCol);
      lldNgNu1DCol = std::max(nrowsNgNu1DCol, 1);

      nrowsNgNb1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
      ncolsNgNb1DCol = SCALAPACK(numroc)(&Nb, &Basis_BlockSize, &mycol1DCol, &I_ZERO, &npcol1DCol);
      lldNgNb1DCol = std::max(nrowsNgNb1DCol, 1);

      nrowsNuNu1DCol = SCALAPACK(numroc)(&Nu, &Nu, &myrow1DCol, &I_ZERO, &nprow1DCol);
      ncolsNuNu1DCol = SCALAPACK(numroc)(&Nu, &Basis_BlockSize, &mycol1DCol, &I_ZERO, &npcol1DCol);
      lldNuNu1DCol = std::max(nrowsNuNu1DCol, 1);

      SCALAPACK(descinit)(desc_NgNb1DCol, &Ng, &Nb, &Ng, &Basis_BlockSize, &I_ZERO,
                          &I_ZERO, &contxt1DCol, &lldNgNb1DCol, &info1DCol);

      SCALAPACK(descinit)(desc_NgNu1DCol, &Ng, &Nu, &Ng, &Basis_BlockSize, &I_ZERO,
                          &I_ZERO, &contxt1DCol, &lldNgNu1DCol, &info1DCol);

      SCALAPACK(descinit)(desc_NuNu1DCol, &Nu, &Nu, &Nu, &Basis_BlockSize, &I_ZERO,
                          &I_ZERO, &contxt1DCol, &lldNuNu1DCol, &info1DCol);
    }

    nprow1DRow = mpisize;
    npcol1DRow = 1;

    Cblacs_get(0, 0, &contxt1DRow);
    Cblacs_gridmap(&contxt1DRow, &isdf_pmap[0], nprow1DRow, nprow1DRow, npcol1DRow);
    Cblacs_gridinfo(contxt1DRow, &nprow1DRow, &npcol1DRow, &myrow1DRow, &mycol1DRow);

    if (contxt1DRow >= 0)
    {

      nrowsNgNu1DRow = SCALAPACK(numroc)(&Ng, &Grid_BlockSize, &myrow1DRow, &I_ZERO, &nprow1DRow);
      ncolsNgNu1DRow = SCALAPACK(numroc)(&Nu, &Nu, &mycol1DRow, &I_ZERO, &npcol1DRow);
      lldNgNu1DRow = std::max(nrowsNgNu1DRow, 1);

      nrowsNgNb1DRow = SCALAPACK(numroc)(&Ng, &Grid_BlockSize, &myrow1DRow, &I_ZERO, &nprow1DRow);
      ncolsNgNb1DRow = SCALAPACK(numroc)(&Nb, &Nb, &mycol1DRow, &I_ZERO, &npcol1DRow);
      lldNgNb1DRow = std::max(nrowsNgNb1DRow, 1);

      nrowsNbNu1DRow = SCALAPACK(numroc)(&Nb, &Basis_BlockSize, &myrow1DRow, &I_ZERO, &nprow1DRow);
      ncolsNbNu1DRow = SCALAPACK(numroc)(&Nu, &Nu, &mycol1DRow, &I_ZERO, &npcol1DRow);
      lldNbNu1DRow = std::max(nrowsNbNu1DRow, 1);

      SCALAPACK(descinit)(desc_NgNb1DRow, &Ng, &Nb, &Grid_BlockSize, &Nb, &I_ZERO,
                          &I_ZERO, &contxt1DRow, &lldNgNb1DRow, &info1DRow);

      SCALAPACK(descinit)(desc_NgNu1DRow, &Ng, &Nu, &Grid_BlockSize, &Nu, &I_ZERO,
                          &I_ZERO, &contxt1DRow, &lldNgNu1DRow, &info1DRow);

      SCALAPACK(descinit)(desc_NbNu1DRow, &Nb, &Nu, &Basis_BlockSize, &Nu, &I_ZERO,
                          &I_ZERO, &contxt1DRow, &lldNbNu1DRow, &info1DRow);
    }

    double ratio = (double)Ng / (double)Nu;
    double npcol_ideal = std::max(1.0, std::sqrt((double)mpisize / ratio));

    int best_npcol = 1;
    double best_diff = std::abs(npcol_ideal - 1);
    for (int nc = 1; nc <= mpisize; ++nc)
    {
      if (mpisize % nc == 0)
      {
        double diff = std::abs(nc - npcol_ideal);
        if (diff < best_diff)
        {
          best_diff = diff;
          best_npcol = nc;
        }
      }
    }
    npcol2D = best_npcol;
    nprow2D = mpisize / npcol2D;

    if (mpirank == 0)
    {
      isdfOFS << "TallGrid: Ng=" << Ng << ", Nu=" << Nu
              << ", grid=" << nprow2D << " x " << npcol2D << std::endl;
    }

    Cblacs_get(0, 0, &contxt2D);
    Cblacs_gridmap(&contxt2D, &isdf_pmap[0], nprow2D, nprow2D, npcol2D);
    Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);

    Int mb2D = Grid_BlockSize;
    Int nb2D = Grid_BlockSize;
    int nprowSq, npcolSq, myrowSq, mycolSq;
    int contxtSq;

    int npcol_ideal_sq = std::max(1, (int)std::round(std::sqrt((double)mpisize)));

    int best_npcol_sq = 1;
    int best_diff_sq = std::abs(npcol_ideal_sq - 1);
    for (int nc = 1; nc <= mpisize; ++nc)
    {
      if (mpisize % nc == 0)
      {
        int diff = std::abs(nc - npcol_ideal_sq);
        if (diff < best_diff_sq)
        {
          best_diff_sq = diff;
          best_npcol_sq = nc;
        }
      }
    }
    npcolSq = best_npcol_sq;
    nprowSq = mpisize / npcolSq;

    if (mpirank == 0)
    {
      isdfOFS << "SquareGrid: grid=" << nprowSq << " x " << npcolSq << std::endl;
      isdfOFS << "Grid_BlockSize" << Grid_BlockSize << std::endl;
    }

    IntNumVec sq_pmap(mpisize);
    for (Int pmap_iter = 0; pmap_iter < mpisize; pmap_iter++)
    {
      sq_pmap[pmap_iter] = pmap_iter;
    }

    Cblacs_get(0, 0, &contxtSq);
    Cblacs_gridmap(&contxtSq, &sq_pmap[0], nprowSq, nprowSq, npcolSq);
    Cblacs_gridinfo(contxtSq, &nprowSq, &npcolSq, &myrowSq, &mycolSq);

    int blockSizeSq = Grid_BlockSize;


    int nrowsNuNuSq = SCALAPACK(numroc)(&Nu, &blockSizeSq, &myrowSq, &I_ZERO, &nprowSq);
    int ncolsNuNuSq = SCALAPACK(numroc)(&Nu, &blockSizeSq, &mycolSq, &I_ZERO, &npcolSq);
    int lldNuNuSq = std::max(nrowsNuNuSq, 1);

    int desc_NuNuSq[9];
    int info;

    SCALAPACK(descinit)(desc_NuNuSq, &Nu, &Nu, &blockSizeSq, &blockSizeSq,
                    &I_ZERO, &I_ZERO, &contxtSq, &lldNuNuSq, &info);
    if (contxt2D >= 0)
    {

      nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2D, &myrow2D, &I_ZERO, &nprow2D);
      ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
      lldNgNu2D = std::max(nrowsNgNu2D, 1);

      nrowsNgNb2D = SCALAPACK(numroc)(&Ng, &mb2D, &myrow2D, &I_ZERO, &nprow2D);
      ncolsNgNb2D = SCALAPACK(numroc)(&Nb, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
      lldNgNb2D = std::max(nrowsNgNb2D, 1);

      nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &myrow2D, &I_ZERO, &nprow2D);
      ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
      lldNuNu2D = std::max(nrowsNuNu2D, 1);

      nrowsNbNu2D = SCALAPACK(numroc)(&Nb, &nb2D, &myrow2D, &I_ZERO, &nprow2D);
      ncolsNbNu2D = SCALAPACK(numroc)(&Nu, &nb2D, &mycol2D, &I_ZERO, &npcol2D);
      lldNbNu2D = std::max(nrowsNbNu2D, 1);

      SCALAPACK(descinit)(desc_NgNb2D, &Ng, &Nb, &mb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNgNb2D, &info2D);

      SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNgNu2D, &info2D);

      SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &nb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNuNu2D, &info2D);

      SCALAPACK(descinit)(desc_NbNu2D, &Nb, &Nu, &nb2D, &nb2D, &I_ZERO,
                          &I_ZERO, &contxt2D, &lldNbNu2D, &info2D);
    }

    IntNumVec pivMu1(Nu);
//    GetTime(timeSta2);
    GetTime(timeSta1);
    for (Int mu = 0; mu < Nu; mu++)
    {
      pivMu1(mu) = pivQR_(mu);
    }

    Int NbLocal = Nb / mpisize;
    if (mpirank < (Nb % mpisize))
    {
      NbLocal++;
    }

    DblNumMat psiMuRow(NbLocal, Nu);
    SetValue(psiMuRow, 0.0);

    for (Int k = 0; k < NbLocal; k++)
    {
      for (Int mu = 0; mu < Nu; mu++)
      {
        psiMuRow(k, mu) = psiCol(pivMu1(mu), k);
      }
    }

    DblNumMat psiMu2D(nrowsNbNu2D, ncolsNbNu2D);

    SetValue(psiMu2D, 0.0);
    GetTime(timeStaMPI);

    SCALAPACK(pdgemr2d)(&Nb, &Nu, psiMuRow.Data(), &I_ONE, &I_ONE, desc_NbNu1DRow,
                        psiMu2D.Data(), &I_ONE, &I_ONE, desc_NbNu2D, &contxt1DRow);
    GetTime(timeEndMPI);
    timeMPI = timeMPI + timeEndMPI - timeStaMPI;

    GetTime(timeEnd1);

    isdfOFS << "Time for computing psiMu2D is " << timeEnd1 - timeSta1 << " [s]" << std::endl
            << std::endl;
    psiMuRow.FreeData();

    DblNumMat psi2D(nrowsNgNb2D, ncolsNgNb2D);

    SetValue(psi2D, 0.0);

    GetTime(timeStaMPI);
    SCALAPACK(pdgemr2d)(&Ng, &Nb, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNb1DCol,
                        psi2D.Data(), &I_ONE, &I_ONE, desc_NgNb2D, &contxt1DCol);
    GetTime(timeEndMPI);
    timeMPI = timeMPI + timeEndMPI - timeStaMPI;

    DblNumMat PpsiMu2D(nrowsNgNu2D, ncolsNgNu2D);
    SetValue(PpsiMu2D, 0.0);
    Int Nstart1 = nocc - nv1 + 1;
    Int K_size1 = nv1 + nc1;
    GetTime(timeSta2);

    GetTime(timeStaGemm);
    my_pdgemm("N", "N", &Ng, &Nu, &K_size1,
              &D_ONE,
              psi2D.Data(), &I_ONE, &Nstart1, desc_NgNb2D,
              psiMu2D.Data(), &Nstart1, &I_ONE, desc_NbNu2D,
              &D_ZERO,
              PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    GetTime(timeEndGemm);

    timeGemm = timeGemm + timeEndGemm - timeStaGemm;

    isdfOFS << "nrowsNgNu2D=" << nrowsNgNu2D << "ncolsNgNu2D" << ncolsNgNu2D << std::endl;
    DblNumMat Xi2D(nrowsNgNu2D, ncolsNgNu2D);
    SetValue(Xi2D, 0.0);

    if (nv1 != nv2 || nc1 != nc2)
    {

      Int Nstart2 = nocc - nv2 + 1;
      Int K_size2 = nv2 + nc2;
      GetTime(timeStaGemm);



      my_pdgemm("N", "N", &Ng, &Nu, &K_size2,
                &D_ONE,
                psi2D.Data(), &I_ONE, &Nstart2, desc_NgNb2D,
                psiMu2D.Data(), &Nstart2, &I_ONE, desc_NbNu2D,
                &D_ZERO,
                Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

      GetTime(timeEndGemm);
      timeGemm = timeGemm + timeEndGemm - timeStaGemm;
      psi2D.FreeData();
      psiMu2D.FreeData();
      GetTime(timeEnd2);
      isdfOFS << "Time for Pphimu and Pphimu  is " << timeEnd2 - timeSta2 << " [s]" << std::endl;

      GetTime(timeSta2);
      Real *Xi2DPtr = Xi2D.Data();
      Real *PpsiMu2DPtr = PpsiMu2D.Data();

      for (Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++)
      {
        Xi2DPtr[g] = PpsiMu2DPtr[g] * Xi2DPtr[g];
      }
      PpsiMu2D.FreeData();

    }
    else
    {
      psi2D.FreeData();
      psiMu2D.FreeData();
      GetTime(timeEnd2);
      isdfOFS << "Time for Pphimu and Pphimu  is " << timeEnd2 - timeSta2 << " [s]" << std::endl;
      GetTime(timeSta2);
      DblNumMat &PphiMu2D = PpsiMu2D;
      Real *Xi2DPtr = Xi2D.Data();
      Real *PpsiMu2DPtr = PpsiMu2D.Data();
      Real *PphiMu2DPtr = PphiMu2D.Data();
      for (Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++)
      {
        Xi2DPtr[g] = PpsiMu2DPtr[g] * PphiMu2DPtr[g];
      }
      PpsiMu2D.FreeData();
    }
    GetTime(timeEnd2);

    isdfOFS << "Time for Z*C^T  is " << timeEnd2 - timeSta2 << " [s]" << std::endl;

    GetTime(timeSta1);

    isdfOFS << "nrowsNgNu1DCol=" << nrowsNgNu1DCol << "ncolsNgNu1DCol" << ncolsNgNu1DCol << std::endl;
    DblNumMat Xi1D(nrowsNgNu1DCol, ncolsNgNu1DCol);
    SetValue(Xi1D, 0.0);
    GetTime(timeStaMPI);

    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
                        Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol, &contxt2D);
    GetTime(timeEndMPI);
    timeMPI = timeMPI + timeEndMPI - timeStaMPI;

    isdfOFS << "nrowsNuNu1DCol=" << nrowsNuNu1DCol << "ncolsNuNu1DCol" << ncolsNuNu1DCol << std::endl;
    DblNumMat PMuNu1D(nrowsNuNu1DCol, ncolsNuNu1DCol);
    SetValue(PMuNu1D, 0.0);

    for (Int mu = 0; mu < nrowsNuNu1DCol; mu++)
    {
      for (Int nu = 0; nu < ncolsNuNu1DCol; nu++)
      {
        PMuNu1D(mu, nu) = Xi1D(pivMu1(mu), nu);
      }
    }

    double local_trace = 0.0;
    Int nb = Basis_BlockSize;
    Int npcol = npcol1DCol;    
    Int mycol = mycol1DCol;    

    for (Int nu = 0; nu < ncolsNuNu1DCol; nu++)
    {

      Int global_nu = (nu / nb) * (npcol * nb) + (mycol * nb) + (nu % nb);

      if (global_nu < nrowsNuNu1DCol)
      {
        local_trace += PMuNu1D(global_nu, nu);
      }
    }
    double global_trace = 0.0;
    GetTime(timeStaMPI);
    MPI_Allreduce(&local_trace, &global_trace, 1, MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI = timeMPI + timeEndMPI - timeStaMPI;
    double eps = 1e-10 * global_trace / (double)Nu;

    for (Int nu = 0; nu < ncolsNuNu1DCol; nu++)
    {
      Int global_nu = (nu / nb) * (npcol * nb) + (mycol * nb) + (nu % nb);

      if (global_nu < nrowsNuNu1DCol)
      {
        PMuNu1D(global_nu, nu) += eps;
      }
    }

    Xi1D.FreeData();


    pivMu1.FreeData();
    isdfOFS << "nrowsNuNuSq=" << nrowsNuNuSq << "ncolsNuNuSq=" << ncolsNuNuSq << std::endl;
    DblNumMat PMuNu2D_sq(nrowsNuNuSq, ncolsNuNuSq);
    GetTime(timeStaMPI);
    SCALAPACK(pdgemr2d)(&Nu, &Nu,
                        PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1DCol,
                        PMuNu2D_sq.Data(), &I_ONE, &I_ONE, desc_NuNuSq,
                        &contxt1DCol);
    GetTime(timeEndMPI);

    timeMPI = timeMPI + timeEndMPI - timeStaMPI;
    GetTime(timeEnd1);
    PMuNu1D.FreeData();

    isdfOFS << "Time for C*C^T is " << timeEnd1 - timeSta1 << " [s]" << std::endl
            << std::endl;
    if (1)
    {
      DblNumMat PMuNu2D(nrowsNuNu2D, ncolsNuNu2D);
      SetValue(PMuNu2D, 0.0);

      GetTime(timeSta1);
      SCALAPACK(pdpotrf)("L", &Nu, PMuNu2D_sq.Data(), &I_ONE, &I_ONE, desc_NuNuSq, &info);
      GetTime(timeEnd1);
      isdfOFS << "Time for PMuNu Potrf solve is " << timeEnd1 - timeSta1 << " [s]" << std::endl;
      if (info != 0)
      {
        ErrorHandling("Matrix is not positive definite.");
      }
      GetTime(timeStaMPI);
      SCALAPACK(pdgemr2d)(&Nu, &Nu, PMuNu2D_sq.Data(), &I_ONE, &I_ONE, desc_NuNuSq,
                          PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt2D);
      GetTime(timeEndMPI);
      timeMPI = timeMPI + timeEndMPI - timeStaMPI;
      PMuNu2D_sq.FreeData();
      GetTime(timeSta1);
      SCALAPACK(pdtrsm)("R", "L", "T", "N",
                        &Ng, &Nu,
                        &D_ONE,
                        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
                        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
      SCALAPACK(pdtrsm)("R", "L", "N", "N",
                        &Ng, &Nu,
                        &D_ONE,
                        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
                        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
      GetTime(timeEnd1);
      isdfOFS << "Time for TRSM solve is " << timeEnd1 - timeSta1 << " [s]" << std::endl;
    }
    isdfOFS << "Time for Gemm  in ISDF is " << timeGemm << " [s]" << std::endl;
    GetTime(timeStaMPI);
    SCALAPACK(pdgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
                        Xi.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol, &contxt2D);
    GetTime(timeEndMPI);
    timeMPI = timeMPI + timeEndMPI - timeStaMPI;
    isdfOFS << "Time for MPI in getbasis is " << timeMPI << " [s]" << std::endl;
    timeISDFMPI = timeISDFMPI + timeMPI;
//    timeISDFGEMM = timeISDFGEMM + timeGemm;
    if (contxt1DCol >= 0)
    {
      Cblacs_gridexit(contxt1DCol);
    }

    if (contxt1DRow >= 0)
    {
      Cblacs_gridexit(contxt1DRow);
    }

    if (contxt2D >= 0)
    {
      Cblacs_gridexit(contxt2D);
    }
    if (contxtSq >= 0)
    {
      Cblacs_gridexit(contxtSq);
    }
    return;
  }
#endif
}
