#pragma once

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
namespace isdf{

struct Fourier {
  Domain           domain;
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridTotalFine;
  // plans
  fftw_plan backwardPlan;
  fftw_plan forwardPlan;
//  fftw_plan backwardPlanFine;
//  fftw_plan forwardPlanFine;
  fftw_plan backwardPlanR2C;
  fftw_plan forwardPlanR2C;
//  fftw_plan backwardPlanR2CFine;
//  fftw_plan forwardPlanR2CFine;
  unsigned  plannerFlag;

  // Laplacian operator related
  DblNumVec                gkk;
  std::vector<CpxNumVec>   ik;
  DblNumVec                TeterPrecond;

//  DblNumVec                gkkFine;
//  std::vector<CpxNumVec>   ikFine;
  // FIXME Teter should be moved to Hamiltonian
//  DblNumVec                TeterPrecondFine;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVec;     
  CpxNumVec                outputComplexVec;     

//  CpxNumVec                inputComplexVecFine;     
//  CpxNumVec                outputComplexVecFine;     

  // Real data Fourier transform
  Int       numGridTotalR2C;
//  Int       numGridTotalR2CFine;

  DblNumVec                gkkR2C;
  std::vector<CpxNumVec>   ikR2C;
  DblNumVec                TeterPrecondR2C;

//  DblNumVec                gkkR2CFine;
//  std::vector<CpxNumVec>   ikR2CFine;
//  DblNumVec                TeterPrecondR2CFine;

  // Temporary vectors that can also be used globally
  DblNumVec                inputVecR2C;     
  CpxNumVec                outputVecR2C;     

//  DblNumVec                inputVecR2CFine;     
//  CpxNumVec                outputVecR2CFine;     

  /// @brief index array for mapping a coarse grid to a fine grid
  IntNumVec                idxFineGrid;
  IntNumVec                idxFineGridR2C;


  Fourier();
  ~Fourier();

  void Initialize( const Domain& dm );
//  void InitializeFine( const Domain& dm );

};
void FFTWExecute( Fourier& fft, fftw_plan& plan );

/*
struct DistFourier {
  Domain           domain;                      
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridLocal;    
  ptrdiff_t        localNz;
  ptrdiff_t        localNzStart;
  // numAllocLocal is the size for the FFTW vectors.
  // numAllocLocal is close but may not be exactly the same as numGridLocal, which is
  // localNz * Ny * Nx.  This is because FFTW may need some intermediate
  // space.  For more information see FFTW's manual.
  ptrdiff_t        numAllocLocal;

  /// @brief Whether the processor according to the rank in domain
  /// participate in the FFTW calculation.
  bool             isInGrid;
  /// @brief The communicator used by parallel FFTW, should be
  /// consistent with inGrid.
  MPI_Comm         comm;

  // plans
  fftw_plan        backwardPlan;
  fftw_plan        forwardPlan;

  unsigned         plannerFlag;

  // Laplacian operator related
  DblNumVec                gkkLocal;
  std::vector<CpxNumVec>   ikLocal;
  DblNumVec                TeterPrecondLocal;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVecLocal;     
  CpxNumVec                outputComplexVecLocal;     


  void Initialize( const Domain& dm, Int numProc );
};
*/
} // namespace isdf


