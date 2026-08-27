#include "../../include/isdf.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace isdf{
#ifdef  GPU

void isdffunc::ISDF_GPU(cuDblNumMat &cu_psiRow, Domain &domain_,
                        const Int &nv_, const Int &nv1, const Int &nc1,
                        const Int &nv2, const Int &nc2,
                        Int &ntotLocal, Int rk,
                        cuDblNumMat &cu_PphimuLocal, IntNumVec &pivQR_,
                        Int &KmeansMaxIter_ISDF,Real &timeISDFMemcpy,Real &timeISDFMPI, bool check)
{
  Real timeSta, timeStaisdf, timeStaMPI, timeStabasis;
  Real timeEnd, timeEndisdf, timeEndMPI, timeEndbasis;
  Real timeStaCopy, timeEndCopy;
  Real timeStaGemm, timeEndGemm;
  Real timeMPI = 0.0;
  Real timeMPIKmeans = 0.0;
  Real timeGemm = 0.0;
  Real timeCopy = 0.0;
  MPI_Barrier(domain_.comm);
  GetTime(timeStaisdf);
//  MPI_Barrier(domain_.comm);
  int mpirank, mpisize;
  MPI_Comm_rank(domain_.comm, &mpirank);
  MPI_Comm_size(domain_.comm, &mpisize);

  Real KmeansTolerance = 1e-5;
  Real Tolerance = 1e-8;

  double d_one = 1.0;
  double d_zero = 0.0;
  Int Ng = domain_.NumGridTotal();

  cusolver::Init();
  cublas::Init();

  Int psiRow_m = cu_psiRow.m();
  Int psiRow_n = cu_psiRow.n();

  Int nstateLocal = psiRow_n / mpisize;
  if (mpirank < (psiRow_n % mpisize)) nstateLocal++;
  {
    GetTime(timeSta);
    IntNumVec weightSizeDispls(mpisize);
    IntNumVec weightSize(mpisize);
    DblNumVec weightLocal(ntotLocal);
    cuDblNumVec cu_weightLocal(ntotLocal);
    SetValue(weightLocal, 0.0);
    DblNumVec weight(Ng);
    SetValue(weight, 0.0);

    Int ntotBlocksize = Ng / mpisize;

    calculateKmeansWeight(cu_psiRow.Data(), ntotLocal,
                          nv_ - nv1, nv_ + nc1,
                          nv_ - nv2, nv_ + nc2,
                          cu_weightLocal.Data());

    GetTime(timeStaCopy);
    cuda_memcpy_GPU2CPU(weightLocal.Data(), cu_weightLocal.Data(),
                        ntotLocal * sizeof(double));
    GetTime(timeEndCopy);
    timeCopy += timeEndCopy - timeStaCopy;
    cu_weightLocal.FreeData();   

    if ((Ng % mpisize) == 0) {
      for (Int i = 0; i < mpisize; i++) {
        weightSizeDispls[i] = i * ntotBlocksize;
        weightSize[i] = ntotBlocksize;
      }
    } else {
      for (Int i = 0; i < mpisize; i++) {
        if (i < (Ng % mpisize)) {
          weightSizeDispls[i] = i * (ntotBlocksize + 1);
          weightSize[i] = ntotBlocksize + 1;
        } else {
          weightSizeDispls[i] = (Ng % mpisize) * (ntotBlocksize + 1)
                              + (i - (Ng % mpisize)) * ntotBlocksize;
          weightSize[i] = ntotBlocksize;
        }
      }
    }
//    isdfOFS<<"ntot_"<<Ng<<" "<<weight.m()<<std::endl;
    cudaDeviceSynchronize();
    MPI_Barrier(domain_.comm); 
    GetTime(timeStaMPI);
    MPI_Allgatherv(weightLocal.Data(), ntotLocal, MPI_DOUBLE,
                   weight.Data(), weightSize.Data(), weightSizeDispls.Data(),
                   MPI_DOUBLE, domain_.comm);
    GetTime(timeEndMPI);
    timeMPIKmeans += timeEndMPI - timeStaMPI;
    //printCpxV(weight,isdfOFS);
    weightLocal.FreeData();   

    KMEAN(Ng, weight, rk, KmeansTolerance, KmeansMaxIter_ISDF, Tolerance,
          domain_, pivQR_.Data());

    weight.FreeData();        
    

//    printCpxV(pivQR_, psiOFS);

    GetTime(timeEnd);
    isdfOFS << "Time for pivQR_ with Kmeans      = " << timeEnd - timeSta << " [s]" << std::endl;
    isdfOFS << "Time for MPI in  Kmeans          = " << timeMPIKmeans << " [s]" << std::endl;
  }

  {
    cuDblNumMat cu_PMuNu;

    GetTime(timeStabasis);
//    GetTime(timeSta);

    Int ntotBlocksize = Ng / mpisize;
    IntNumVec ntotresTotal(mpisize);
    SetValue(ntotresTotal, 0);
    if ((Ng % mpisize) == 0) {
      for (int i = 0; i < mpisize; i++) ntotresTotal(i) = i * ntotBlocksize;
    } else {
      for (Int i = 0; i < mpisize; i++) {
        if (i < (Ng % mpisize))
          ntotresTotal(i) = i * (ntotBlocksize + 1);
        else
          ntotresTotal(i) = (Ng % mpisize) * (ntotBlocksize + 1)
                          + (i - (Ng % mpisize)) * ntotBlocksize;
      }
    }


    Int rkLocal = rk / mpisize;
    if (mpirank < (rk % mpisize)) rkLocal++;



    DblNumMat psiRow(psiRow_m, psiRow_n);
    GetTime(timeStaCopy);
    cuda_memcpy_GPU2CPU(psiRow.Data(), cu_psiRow.Data(),
                      psiRow_m * psiRow_n * sizeof(double));
    GetTime(timeEndCopy);
    timeCopy += timeEndCopy - timeStaCopy;

    DblNumMat psiCol(Ng, nstateLocal);
    SetValue(psiCol, 0.0);

    GetTime(timeStaMPI);
    AlltoallBackward(psiRow, psiCol, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI += timeEndMPI - timeStaMPI;


    DblNumMat psiMuLocalCol(rk, psiCol.n());
    SetValue(psiMuLocalCol, 0.0);
    for (int i = 0; i < psiCol.n(); i++)
      for (int j = 0; j < rk; j++)
        psiMuLocalCol(j, i) = psiCol(pivQR_[j], i);

    psiCol.FreeData();   

    DblNumMat psiMuLocalRow(rkLocal, psiRow_n);
    SetValue(psiMuLocalRow, 0.0);
    GetTime(timeStaMPI);
    AlltoallForward(psiMuLocalCol, psiMuLocalRow, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI += timeEndMPI - timeStaMPI;

    psiMuLocalCol.FreeData();   

    DblNumMat psiMuLocalRowT(psiRow_n, rkLocal);
    for (int i = 0; i < psiRow_n; i++)
      for (int j = 0; j < rkLocal; j++)
        psiMuLocalRowT(i, j) = psiMuLocalRow(j, i);

    psiMuLocalRow.FreeData();  

    IntNumVec widthLocals(mpisize);
    GetTime(timeStaMPI);
    MPI_Allgather(&rkLocal, 1, MPI_INT, widthLocals.Data(), 1, MPI_INT,
                  domain_.comm);
    GetTime(timeEndMPI);
    timeMPI += timeEndMPI - timeStaMPI;

    DblNumMat psiMuRow(psiRow_n, rk);
    SetValue(psiMuRow, 0.0);
    IntNumVec sendcountsum(mpisize);
    IntNumVec displspsi(mpisize);
    SetValue(sendcountsum, 0);
    SetValue(displspsi, 0);
    for (int i = 0; i < mpisize; i++)
      sendcountsum[i] = widthLocals[i] * psiRow_n;
    for (int i = 1; i < mpisize; i++)
      displspsi[i] = displspsi[i - 1] + widthLocals[i - 1] * psiRow_n;

    GetTime(timeStaMPI);
    MPI_Allgatherv(psiMuLocalRowT.Data(), rkLocal * psiRow_n, MPI_DOUBLE,
                   psiMuRow.Data(), sendcountsum.Data(), displspsi.Data(),
                   MPI_DOUBLE, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI += timeEndMPI - timeStaMPI;

    psiMuLocalRowT.FreeData();   

    cuDblNumMat cu_psiMuRow(psiRow_n, rk);
    GetTime(timeStaCopy);
    cuda_memcpy_CPU2GPU(cu_psiMuRow.Data(), psiMuRow.Data(),
                        psiRow_n * rk * sizeof(double));
    GetTime(timeEndCopy);
    timeCopy += timeEndCopy - timeStaCopy;

    psiMuRow.FreeData();   

   
    cuDblNumMat cu_PpsimuLocal(ntotLocal, rk);
    cudaDeviceSynchronize();
    GetTime(timeSta);
    GetTime(timeStaGemm);
    cublas::Gemm(CUBLAS_OP_N, CUBLAS_OP_N,
                 ntotLocal, rk, (nv1 + nc1),
                 &d_one,
                 cu_psiRow.Data() + (nv_ - nv1) * ntotLocal, ntotLocal,
                 cu_psiMuRow.Data() + (nv_ - nv1), psiRow_n,
                 &d_zero,
                 cu_PpsimuLocal.Data(), ntotLocal);
    cudaDeviceSynchronize();
    GetTime(timeEndGemm);
    timeGemm += timeEndGemm - timeStaGemm;

    if (nv1 != nv2 || nc1 != nc2) {

      cudaDeviceSynchronize();
      GetTime(timeStaGemm);
      cublas::Gemm(CUBLAS_OP_N, CUBLAS_OP_N,
                   ntotLocal, rk, (nv2 + nc2),
                   &d_one,
                   cu_psiRow.Data() + (nv_ - nv2) * ntotLocal, ntotLocal,
                   cu_psiMuRow.Data() + (nv_ - nv2), psiRow_n,
                   &d_zero,
                   cu_PphimuLocal.Data(), ntotLocal);
      cudaDeviceSynchronize();
      GetTime(timeEndGemm);
      timeGemm += timeEndGemm - timeStaGemm;

      cu_psiMuRow.FreeData();   

      GetTime(timeEnd);
      isdfOFS << "Time for Pphimu and Ppsimu       = " << timeEnd - timeSta << " [s]" << std::endl;
      cudaDeviceSynchronize();
      GetTime(timeSta);
      // psiphizeta = Ppsimu .* Pphimu
      getelemproduct(rk, ntotLocal,
                     cu_PphimuLocal.Data(),
                     cu_PpsimuLocal.Data(),
                     cu_PphimuLocal.Data());
      cudaDeviceSynchronize();
      GetTime(timeEnd);
      isdfOFS << "Time for Z*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;
    } else {
      cu_psiMuRow.FreeData();  

      GetTime(timeEnd);
      isdfOFS << "Time for Pphimu and Ppsimu       = " << timeEnd - timeSta << " [s]" << std::endl;
      cudaDeviceSynchronize();
      GetTime(timeSta);
      getelemproduct(rk, ntotLocal,
                     cu_PphimuLocal.Data(),
                     cu_PpsimuLocal.Data(),
                     cu_PpsimuLocal.Data());
      cudaDeviceSynchronize();
      GetTime(timeEnd);
      isdfOFS << "Time for Z*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;
    }
    cu_PpsimuLocal.FreeData(); 


    //printCpxM(cu_PphimuLocal,psiOFS);
    GetTime(timeSta);

    std::vector<Int> ownedNu;
    std::vector<Int> ownedLocalIdx;
    ownedNu.reserve(rk);
    ownedLocalIdx.reserve(rk);
    for (Int nu = 0; nu < rk; nu++) {
      Int globalIdx = pivQR_(nu);
      bool owned;
      if (mpirank < mpisize - 1) {
        owned = (globalIdx >= ntotresTotal(mpirank)) &&
                (globalIdx <  ntotresTotal(mpirank + 1));
      } else {
        owned = (globalIdx >= ntotresTotal(mpirank)) &&
                (globalIdx <  Ng);
      }
      if (owned) {
        ownedNu.push_back(nu);
        ownedLocalIdx.push_back(globalIdx - ntotresTotal(mpirank));
      }
    }
    Int selectedRow = (Int)ownedNu.size();

    DblNumMat PMuNu(rk, rk);
    SetValue(PMuNu, 0.0);

    if (selectedRow > 0) {
      cuIntNumVec cu_localRowIdx(selectedRow);
      GetTime(timeStaCopy);
      cuda_memcpy_CPU2GPU(cu_localRowIdx.Data(), ownedLocalIdx.data(),
                          selectedRow * sizeof(int));
      GetTime(timeEndCopy);
      timeCopy += timeEndCopy - timeStaCopy;

      cuDblNumMat cu_PMuNu_Local(rk, selectedRow);
      getpsiMuT(cu_PphimuLocal.Data(), cu_localRowIdx.Data(),
                cu_PMuNu_Local.Data(), selectedRow, ntotLocal, rk, 0);
      cudaDeviceSynchronize(); 
      cu_localRowIdx.FreeData();   

      DblNumMat PMuNu_Local(rk, selectedRow);
      GetTime(timeStaCopy);
      cuda_memcpy_GPU2CPU(PMuNu_Local.Data(), cu_PMuNu_Local.Data(),
                          rk * selectedRow * sizeof(double));
      GetTime(timeEndCopy);
      timeCopy += timeEndCopy - timeStaCopy;
      cu_PMuNu_Local.FreeData();   

      for (Int k = 0; k < selectedRow; k++) {
        Int nu = ownedNu[k];
        for (Int mu = 0; mu < rk; mu++) {
          PMuNu(mu, nu) = PMuNu_Local(mu, k);
        }
      }
      PMuNu_Local.FreeData();   
    }

    GetTime(timeStaMPI);
    MPI_Allreduce(MPI_IN_PLACE, PMuNu.Data(), rk * rk,
                  MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEndMPI);
    timeMPI += timeEndMPI - timeStaMPI;
//    printCpxM(cu_PphimuLocal,psiOFS);
    GetTime(timeEnd);
    isdfOFS << "Time for C*C^T                   = " << timeEnd - timeSta << " [s]" << std::endl;

 //   GetTime(timeSta);
    double traceP = 0.0;
    for (int i = 0; i < rk; i++) traceP += PMuNu(i, i);
    double eps = 1e-10 * (traceP / double(rk));
    for (int i = 0; i < rk; i++) PMuNu(i, i) += eps;

    //lapack::Potrf('L', rk, PMuNu.Data(), rk);
    //printCpxM(PMuNu,psiOFS);
    cu_PMuNu.Resize(rk, rk);
    GetTime(timeStaCopy);
    cuda_memcpy_CPU2GPU(cu_PMuNu.Data(), PMuNu.Data(),
                        rk * rk * sizeof(double));
    GetTime(timeEndCopy);
    timeCopy += timeEndCopy - timeStaCopy;
    PMuNu.FreeData();

 
    //printCpxM(cu_PMuNu,psiOFS);
    cudaDeviceSynchronize();
    GetTime(timeSta);
    cusolver::Potrf('L', rk, cu_PMuNu.Data(), rk);
    cudaDeviceSynchronize();
    //printCpxM(cu_PMuNu,psiOFS);
    GetTime(timeEnd);
    isdfOFS << "Time for Potrf                   = " << timeEnd - timeSta << " [s]" << std::endl;
    cudaDeviceSynchronize();
    GetTime(timeSta);
    cublas::Trsm(CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                 ntotLocal, rk, &d_one,
                 cu_PMuNu.Data(), rk,
                 cu_PphimuLocal.Data(), ntotLocal);
    //printCpxM(cu_PphimuLocal,psiOFS);
    cublas::Trsm(CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                 ntotLocal, rk, &d_one,
                 cu_PMuNu.Data(), rk,
                 cu_PphimuLocal.Data(), ntotLocal);
    cudaDeviceSynchronize();
    cu_PMuNu.FreeData();   
   // printCpxM(cu_PphimuLocal,psiOFS);
    //MPI_Barrier(domain_.comm);
    GetTime(timeEnd);
    GetTime(timeEndbasis);
    GetTime(timeEndisdf);
    timeISDFMemcpy=timeISDFMemcpy+timeCopy;
    timeISDFMPI = timeISDFMPI + timeMPI+timeMPIKmeans;
    isdfOFS << "Time for Trsm                    = " << timeEnd - timeSta << " [s]" << std::endl;
    isdfOFS << "Time for getbasis                = " << timeEndbasis - timeStabasis << " [s]" << std::endl;
    isdfOFS << "ISDF Time for GEMM in get basis  = " << timeGemm << " [s]" << std::endl;
    isdfOFS << "ISDF Time for MPI in  get basis  = " << timeMPI  << " [s]" << std::endl;
 //   isdfOFS << "ISDF Time for CPU-GPU            = " << timeCopy << " [s]" << std::endl;
//    isdfOFS << "Time for ISDF                    = " << timeEndisdf - timeStaisdf << " [s]" << std::endl;
//    isdfOFS << "ISDF Time for MPI in  ISDF       = " << timeMPI + timeMPIKmeans << " [s]" << std::endl;

    cublas::Destroy();
    cusolver::Destroy();
  }
}

#endif
}
