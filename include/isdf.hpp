#pragma once
#include "./common/environment.hpp"
#include "./common/domain.hpp"
#include "./common/nummat_impl.hpp"
#include "./common/numvec_impl.hpp"
#include "./common/fourier.hpp"
//#include "./common/domain.hpp"
#include "./common/utility.hpp"
#include "./cpu/blas.hpp"
#include "./cpu/lapack.hpp"
#include "./cpu/scalapack.hpp"
#ifdef GPU
#include "../include/cuda/cublas.hpp"
#include "../include/cuda/cusolver.hpp"
#include "./cuda/cu_nummat.hpp"
#include "./cuda/cu_numvec_decl.hpp"
#endif

namespace isdf {


namespace isdffunc {
    void FFTR2C(Fourier &fft, DblNumMat &psiphi, CpxNumMat &PsiPhi);
    void FFTC2C(Fourier &fft, DblNumMat &psiphi, CpxNumMat &PsiPhi);


    void getbasis_MPI(isdf::DblNumMat &psiRow,isdf::DblNumMat &psiCol, isdf::Domain &domain_, const isdf::Int &nv_, const isdf::Int &nv1, const isdf::Int &nc1, const isdf::Int &nv2, const isdf::Int &nc2, isdf::Int &ntotLocal, isdf::Int rk, isdf::DblNumMat &psiphizetaRow, isdf::IntNumVec &pivQR_,isdf::Real &timeISDFMPI);

    void getbasis_scalapack(DblNumMat &psiCol, Domain &domain_, const Int &Nstate, const Int &nocc,const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int Nu,DblNumMat &Xi, IntNumVec &pivQR_,Int &scalblocksize, Real &timeISDFMPI);
    void checkcode(DblNumMat &psiRow,DblNumMat &psiCol, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, DblNumMat &psiphizetaRow, IntNumVec &pivQR_, Fourier &fft);

    void getpoints_Kmeans_MPI(DblNumMat &psiRow, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, IntNumVec &pivQR_, Real &timeISDFMPI,Int &KmeansMaxIter_ISDF);
    void getpoints_QRCP(DblNumMat &psiRow, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, IntNumVec &pivQR_);


#if defined(GPU)
    void ISDF(double *d_psirow, Domain domain, int nocc, int nstate, int nv1, int nc1, int nv2,int nc2, int mu_points, double *d_thetaCol, int *piv,int KmeansMaxIter_ISDF,bool check=false);
#else
    void ISDF(double *psirow, Domain domain, int nocc, int nstate, int nv1, int nc1, int nv2,int nc2, int mu_points, double *thetaCol, int *piv, int KmeansMaxIter_ISDF,int scalblocksize=64, std::string s = "scalapack", std::string s1 = "Kmeans",bool check=false);
#endif

#if defined(GPU)
    void ISDF_GPU(cuDblNumMat &cu_psiRow, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1,
                const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, cuDblNumMat &cu_psiphizetaRow,
                  IntNumVec &pivQR_,Int &KmeansMaxIter_ISDF,Real &timeISDFMemcpy, Real &timeISDFMPI,bool check=false);
#endif
} // namespace isdffunc

} // namespace isdf

