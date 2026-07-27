#include "../../include/isdf.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
namespace isdf
{
  void isdffunc::FFTR2C(Fourier &fft, DblNumMat &psiphi, CpxNumMat &PsiPhi)
  {
    Int nr = psiphi.m();
    Int nb = psiphi.n();
    CpxNumVec temp(nr);
    Int idx, idx1;
    Int n2 = fft.domain.numGrid[0];
    Int n1 = fft.domain.numGrid[1];
    Int n0 = fft.domain.numGrid[2];

    for (Int mu = 0; mu < nb; mu++)
    {

      SetValue(fft.inputVecR2C, 0.0);
      SetValue(fft.outputVecR2C, Z_ZERO);
      blas::Copy(nr, psiphi.VecData(mu), 1, fft.inputVecR2C.Data(), 1);
      FFTWExecute(fft, fft.forwardPlanR2C);
      SetValue(temp, Z_ZERO);
      for (Int k = 0; k < n0; k++)
      {
        for (Int j = 0; j < n1; j++)
        {
          for (Int i = 0; i < n2; i++)
          {

            if (i == 0)
            {
              idx = i + j * (n2 / 2 + 1) + k * (n2 / 2 + 1) * n1;
              idx1 = i + j * (n2) + k * (n2)*n1;
              blas::Copy(n2 / 2 + 1, fft.outputVecR2C.Data() + idx, 1, temp.Data() + idx1, 1);
              i = i + n2 / 2;
            }
            else
            {
              idx = i + j * (n2) + k * (n2)*n1;
              idx1 = n2 - i + (n1 - j) % n1 * (n2 / 2 + 1) + (n0 - k) % n0 * (n2 / 2 + 1) * n1;
              *(temp.Data() + idx) = std::conj(*(fft.outputVecR2C.Data() + idx1));
            }
          }
        }
      }
      blas::Copy(nr, temp.Data(), 1, PsiPhi.VecData(mu), 1);
    }
    return;
  }
  void isdffunc::checkcode(DblNumMat &psiRow, DblNumMat &psiCol, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, DblNumMat &psiphizetaRow, IntNumVec &pivQR_, Fourier &fft)
  {

    MPI_Barrier(domain_.comm);
    int mpirank;
    MPI_Comm_rank(domain_.comm, &mpirank);
    Int mpisize;
    MPI_Comm_size(domain_.comm, &mpisize);
    DblNumMat psiphiRow(ntotLocal, (nv1 + nc1) * (nv2 + nc2));

    for (Int k = nv_ - nv1; k < nv_ + nc1; k++)
    {
      for (Int j = nv_ - nv2; j < nv_ + nc2; j++)
      {
        for (Int i = 0; i < ntotLocal; i++)
        {
          psiphiRow(i, (k - (nv_ - nv1)) * (nv2 + nc2) + (j - (nv_ - nv2))) = (psiRow(i, k)) * (psiRow(i, j));
        }
      }
    }

    Int ntot_ = domain_.NumGridTotal();
    DblNumMat psiphiru(rk, (nv1 + nc1) * (nv2 + nc2));
    SetValue(psiphiru, 0.0);
    Int rkLocal = rk / mpisize;
    if (mpirank < (rk % mpisize))
    {
      rkLocal++;
    }

    DblNumMat psiMuLocalCol(rk, psiCol.n());
    for (int i = 0; i < psiCol.n(); i++)
    {
      for (int j = 0; j < rk; j++)
      {
        psiMuLocalCol(j, i) = psiCol(pivQR_[j], i);
      }
    }
    DblNumMat psiMuLocalRow(rkLocal, psiRow.n());
    SetValue(psiMuLocalRow, 0.0);

    AlltoallForward(psiMuLocalCol, psiMuLocalRow, domain_.comm);
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
    MPI_Allgather(&rkLocal, 1, MPI_INT, widthLocals.Data(), 1, MPI_INT, domain_.comm);
    DblNumMat psiMuRow(psiRow.n(), rk);

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

    MPI_Allgatherv(psiMuLocalRowT.Data(), rkLocal * psiRow.n(), MPI_DOUBLE, psiMuRow.Data(), sendcountsum.Data(), displspsi.Data(), MPI_DOUBLE, domain_.comm);

    for (Int k = 0; k < nv1 + nc1; k++)
    {
      for (Int j = 0; j < nv2 + nc2; j++)
      {
        for (Int i = 0; i < rk; i++)
        {

          psiphiru(i, (k) * (nv2 + nc2) + (j)) = (psiMuRow(k + (nv_ - nv1), i)) * (psiMuRow(j + (nv_ - nv2), i));
        }
      }
    }

    psiMuLocalCol.FreeData();
    psiMuLocalRow.FreeData();
    psiMuLocalRowT.FreeData();
    psiMuRow.FreeData();

    DblNumMat psiphirutheta(ntotLocal, (nv2 + nc2) * (nv1 + nc1));
    blas::Gemm('N', 'N', ntotLocal, (nv2 + nc2) * (nv1 + nc1), rk, 1.0, psiphizetaRow.Data(), ntotLocal, psiphiru.Data(), rk, 0.0, psiphirutheta.Data(), ntotLocal);
    double ijerror = 0.0;
    double rhoij = 0.0;
    for (Int j = 0; j < (nv2 + nc2) * (nv1 + nc1); j++)
    {
      for (Int i = 0; i < ntotLocal; i++)
      {
        double diff = psiphiRow(i, j) - psiphirutheta(i, j);
        rhoij += psiphiRow(i, j) * psiphiRow(i, j);
        ijerror += diff * diff;
      }
    }

    double ijerrorsum = 0.0;
    MPI_Allreduce(&ijerror, &ijerrorsum, 1, MPI_DOUBLE, MPI_SUM, domain_.comm);
    double rhoijsum = 0.0;
    MPI_Allreduce(&rhoij, &rhoijsum, 1, MPI_DOUBLE, MPI_SUM, domain_.comm);
    isdfOFS << "sqrt((rhoij-rho^hat)^2) =" << std::sqrt(ijerrorsum) << std::endl;
    isdfOFS << "sqrt(rhoij^2)= " << std::sqrt(rhoijsum) << std::endl;
    isdfOFS << "sqrt((rhoij-rho^hat)^2)/sqrt(rhoij^2)= " << std::sqrt(ijerrorsum) / std::sqrt(rhoijsum) << std::endl;

    Real radius = 5.0;
    DblNumVec Vcoul(ntot_);
    for (Int i = 0; i < ntot_; i++)
    {
      if (fft.gkk(i) < 1e-8)
      {
        Vcoul(i) = 4.0 * PI * radius * radius;
      }
      else
      {

        //    fft.gkk(i) *= 2;
        Vcoul(i) = 8.0 * PI / (2.0 * fft.gkk(i)) * (1 - cos(sqrt(2.0 * fft.gkk(i)) * radius));
      }
    }

    IntNumVec sendcounts(mpisize);
    IntNumVec sendcounts_sum(mpisize);
    for (int i = 0; i < mpisize; i++)
    {
      sendcounts(i) = ntot_ / mpisize;
      if (i < ntot_ % mpisize)
      {
        sendcounts(i)++;
      }
    }
    sendcounts_sum(0) = 0;
    for (int i = 1; i < mpisize; i++)
    {
      sendcounts_sum(i) = sendcounts_sum(i - 1) + sendcounts(i - 1);
    }
    Int num1 = (nv2 + nc2) * (nv1 + nc1) / mpisize;
    if (mpirank < (((nv2 + nc2) * (nv1 + nc1)) % mpisize))
    {
      num1++;
    }
    DblNumMat psiphiCol(ntot_, num1);
    AlltoallBackward(psiphiRow, psiphiCol, domain_.comm);
    psiphiRow.FreeData();
    CpxNumMat PsiPhiCol(ntot_, num1);
    FFTR2C(fft, psiphiCol, PsiPhiCol);
    psiphiCol.FreeData();
    CpxNumMat PsiPhiRow(ntotLocal, ((nv2 + nc2) * (nv1 + nc1)));
    AlltoallForward(PsiPhiCol, PsiPhiRow, domain_.comm);
    PsiPhiCol.FreeData();
    CpxNumMat PsiPhiRowVtemp(ntotLocal, (nv2 + nc2) * (nv1 + nc1));
    for (Int j = 0; j < (nv2 + nc2) * (nv1 + nc1); j++)
    {

      for (Int i = 0; i < ntotLocal; i++)
      {

        PsiPhiRowVtemp(i, j) = PsiPhiRow(i, j) * Vcoul(i + sendcounts_sum(mpirank));
      }
    }

    // printCpxM(sendcounts_sum,psiOFS);

    CpxNumMat ijVkltemp((nv2 + nc2) * (nv1 + nc1), (nv2 + nc2) * (nv1 + nc1));
    blas::Gemm('C', 'N', (nv2 + nc2) * (nv1 + nc1), (nv2 + nc2) * (nv1 + nc1), ntotLocal, 1.0, PsiPhiRow.Data(), ntotLocal, PsiPhiRowVtemp.Data(), ntotLocal, 0.0, ijVkltemp.Data(), (nv2 + nc2) * (nv1 + nc1));
    DblNumMat ijVkltempReal((nv2 + nc2) * (nv1 + nc1), (nv2 + nc2) * (nv1 + nc1));
    for (Int j = 0; j < (nv2 + nc2) * (nv1 + nc1); j++)
    {

      for (Int i = 0; i < (nv2 + nc2) * (nv1 + nc1); i++)
      {

        ijVkltempReal(i, j) = ijVkltemp(i, j).real();
      }
    }
    PsiPhiRowVtemp.FreeData();
    ijVkltemp.FreeData();
    PsiPhiRow.FreeData();
    DblNumMat ijVkl((nv2 + nc2) * (nv1 + nc1), (nv2 + nc2) * (nv1 + nc1));
    MPI_Allreduce(ijVkltempReal.Data(), ijVkl.Data(), (nv2 + nc2) * (nv1 + nc1) * (nv2 + nc2) * (nv1 + nc1), MPI_DOUBLE, MPI_SUM, domain_.comm);
    ijVkltempReal.FreeData();
    Int rkpervc = rk / mpisize;
    if (mpirank < (rk % mpisize))
    {
      rkpervc++;
    }
    DblNumMat psiphirzetaCol(ntot_, rkpervc);
    AlltoallBackward(psiphizetaRow, psiphirzetaCol, domain_.comm);
    // psiphizetaRow.FreeData();
    CpxNumMat vcgzeta_mu(ntot_, rkpervc);

    FFTR2C(fft, psiphirzetaCol, vcgzeta_mu);

    psiphirzetaCol.FreeData();
    CpxNumMat vcgzeta_muRow(ntotLocal, rk);
    AlltoallForward(vcgzeta_mu, vcgzeta_muRow, domain_.comm);

    vcgzeta_mu.FreeData();

    CpxNumMat vcgzetaV(ntotLocal, rk);
    for (Int j = 0; j < rk; j++)
    {

      for (Int i = 0; i < ntotLocal; i++)
      {

        vcgzetaV(i, j) = vcgzeta_muRow(i, j) * Vcoul(i + sendcounts_sum(mpirank));
      }
    }

    CpxNumMat auxcoultemp(rk, rk);

    blas::Gemm('C', 'N', rk, rk, ntotLocal, 1.0, vcgzeta_muRow.Data(), ntotLocal, vcgzetaV.Data(), ntotLocal, 0.0, auxcoultemp.Data(), rk);

    vcgzeta_muRow.FreeData();
    vcgzetaV.FreeData();
    DblNumMat auxcoulRealtemp(rk, rk);
    for (Int j = 0; j < rk; j++)
    {
      for (Int i = 0; i < rk; i++)
      {
        auxcoulRealtemp(i, j) = auxcoultemp(i, j).real();
      }
    }
    DblNumMat auxcoulReal(rk, rk);
    MPI_Allreduce(auxcoulRealtemp.Data(), auxcoulReal.Data(), rk * rk, MPI_DOUBLE, MPI_SUM, domain_.comm);
    auxcoultemp.FreeData();
    auxcoulRealtemp.FreeData();
    DblNumMat Cauxcoul(rk, (nv1 + nc1) * (nv2 + nc2));
    DblNumMat CauxcoulC((nv1 + nc1) * (nv2 + nc2), (nv1 + nc1) * (nv2 + nc2));

    blas::Gemm('N', 'N', rk, (nv1 + nc1) * (nv2 + nc2), rk, 1.0, auxcoulReal.Data(), rk, psiphiru.Data(), rk, 0.0, Cauxcoul.Data(), rk);
    blas::Gemm('T', 'N', (nv1 + nc1) * (nv2 + nc2), (nv1 + nc1) * (nv2 + nc2), rk, 1.0, psiphiru.Data(), rk, Cauxcoul.Data(), rk, 0.0, CauxcoulC.Data(), (nv1 + nc1) * (nv2 + nc2));
    double ijklerror = 0.0;
    double ijklnorm = 0.0;
    for (Int j = 0; j < (nv2 + nc2) * (nv1 + nc1); j++)
    {
      for (Int i = 0; i < (nv2 + nc2) * (nv1 + nc1); i++)
      {
        double diff = ijVkl(i, j) - CauxcoulC(i, j);
        ijklnorm += ijVkl(i, j) * ijVkl(i, j);
        ijklerror += diff * diff;
      }
    }
    //printCpxM(ijVkl, psiOFS);
    // printCpxM(CauxcoulC,psiOFS);

    isdfOFS << "sqrt((ijkl-ijkl^hat)^2)) =" << std::sqrt(ijklerror) << std::endl;
    isdfOFS << "sqrt(ijkl^2) =" << std::sqrt(ijklnorm) << std::endl;
    isdfOFS << "sqrt((ijkl-ijkl^hat)^2)/sqrt(ijkl^2) =" << std::sqrt(ijklerror) / std::sqrt(ijklnorm) << std::endl;
  }
  //}

}
