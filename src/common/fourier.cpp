#include  "../../include/common/fourier.hpp"
#include  "../../include/cpu/blas.hpp"
namespace isdf{



Fourier::Fourier () : 
  isInitialized(false),
  numGridTotal(0),
  plannerFlag(FFTW_MEASURE | FFTW_UNALIGNED )
  {
    backwardPlan  = NULL;
    forwardPlan   = NULL;
    backwardPlanR2C  = NULL;
    forwardPlanR2C   = NULL;
  }

Fourier::~Fourier () 
{
  if( backwardPlan ) fftw_destroy_plan( backwardPlan );
  if( forwardPlan  ) fftw_destroy_plan( forwardPlan );
  if( backwardPlanR2C  ) fftw_destroy_plan( backwardPlanR2C );
  if( forwardPlanR2C   ) fftw_destroy_plan( forwardPlanR2C );
}

void Fourier::Initialize ( const Domain& dm )
{


  if( isInitialized ) {
    ErrorHandling("Fourier has been initialized.");
  }

  domain = dm;
  Index3& numGrid = domain.numGrid;
  Point3& length  = domain.length;

  numGridTotal = domain.NumGridTotal();

  inputComplexVec.Resize( numGridTotal );
  outputComplexVec.Resize( numGridTotal );

  forwardPlan = fftw_plan_dft_3d( 
      numGrid[2], numGrid[1], numGrid[0], 
      reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ), 
      reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
      FFTW_FORWARD, plannerFlag );

  backwardPlan = fftw_plan_dft_3d(
      numGrid[2], numGrid[1], numGrid[0],
      reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
      reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ),
      FFTW_BACKWARD, plannerFlag);

  std::vector<DblNumVec>  KGrid(DIM);                
  for( Int idim = 0; idim < DIM; idim++ ){
    KGrid[idim].Resize( numGrid[idim] );
    for( Int i = 0; i <= numGrid[idim] / 2; i++ ){
      KGrid[idim](i) = i * 2.0 * PI / length[idim];
    }
    for( Int i = numGrid[idim] / 2 + 1; i < numGrid[idim]; i++ ){
      KGrid[idim](i) = ( i - numGrid[idim] ) * 2.0 * PI / length[idim];
    }
  }

  gkk.Resize( dm.NumGridTotal() );
  TeterPrecond.Resize( dm.NumGridTotal() );
  ik.resize(DIM);
  ik[0].Resize( dm.NumGridTotal() );
  ik[1].Resize( dm.NumGridTotal() );
  ik[2].Resize( dm.NumGridTotal() );

  Real*     gkkPtr = gkk.Data();
  Complex*  ikXPtr = ik[0].Data();
  Complex*  ikYPtr = ik[1].Data();
  Complex*  ikZPtr = ik[2].Data();



  for( Int k = 0; k < numGrid[2]; k++ ){
    for( Int j = 0; j < numGrid[1]; j++ ){
      for( Int i = 0; i < numGrid[0]; i++ ){
        *(gkkPtr++) = 
          ( KGrid[0](i) * KGrid[0](i) +
            KGrid[1](j) * KGrid[1](j) +
            KGrid[2](k) * KGrid[2](k) ) / 2.0;

        *(ikXPtr++) = Complex( 0.0, KGrid[0](i) );
        *(ikYPtr++) = Complex( 0.0, KGrid[1](j) );
        *(ikZPtr++) = Complex( 0.0, KGrid[2](k) );

      }
    }
  }

  Real  a, b;
  for( Int i = 0; i < domain.NumGridTotal(); i++ ){
    a = gkk[i] * 2.0;
    b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
    TeterPrecond[i] = b / ( b + 16.0 * pow(a, 4.0) );
  }


  numGridTotalR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];

  inputVecR2C.Resize( numGridTotal );
  outputVecR2C.Resize( numGridTotalR2C );

  forwardPlanR2C = fftw_plan_dft_r2c_3d( 
      numGrid[2], numGrid[1], numGrid[0], 
      ( &inputVecR2C[0] ), 
      reinterpret_cast<fftw_complex*>( &outputVecR2C[0] ),
      plannerFlag );

  backwardPlanR2C = fftw_plan_dft_c2r_3d(
      numGrid[2], numGrid[1], numGrid[0],
      reinterpret_cast<fftw_complex*>( &outputVecR2C[0] ),
      &inputVecR2C[0],
      plannerFlag);
  gkkR2C.Resize( numGridTotalR2C );
  TeterPrecondR2C.Resize( numGridTotalR2C );
  ikR2C.resize(DIM);
  ikR2C[0].Resize( numGridTotalR2C );
  ikR2C[1].Resize( numGridTotalR2C );
  ikR2C[2].Resize( numGridTotalR2C );


  Real*  gkkR2CPtr = gkkR2C.Data();
  Complex*  ikXR2CPtr = ikR2C[0].Data();
  Complex*  ikYR2CPtr = ikR2C[1].Data();
  Complex*  ikZR2CPtr = ikR2C[2].Data();
  for( Int k = 0; k < numGrid[2]; k++ ){
    for( Int j = 0; j < numGrid[1]; j++ ){
      for( Int i = 0; i < numGrid[0]/2+1; i++ ){
        *(gkkR2CPtr++) = 
          ( KGrid[0](i) * KGrid[0](i) +
            KGrid[1](j) * KGrid[1](j) +
            KGrid[2](k) * KGrid[2](k) ) / 2.0;

        *(ikXR2CPtr++) = Complex( 0.0, KGrid[0](i) );
        *(ikYR2CPtr++) = Complex( 0.0, KGrid[1](j) );
        *(ikZR2CPtr++) = Complex( 0.0, KGrid[2](k) );
      }
    }
  }


  for( Int i = 0; i < numGridTotalR2C; i++ ){
    a = gkkR2C[i] * 2.0;
    b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
    TeterPrecondR2C[i] = b / ( b + 16.0 * pow(a, 4.0) );
  }

  isInitialized = true;


  return ;
}        


void FFTWExecute ( Fourier& fft, fftw_plan& plan ){

  Index3& numGrid = fft.domain.numGrid;
  Int ntot      = fft.domain.NumGridTotal();
  Real vol      = fft.domain.Volume();
  Real fac;

  Int ntotR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];

  if ( plan == fft.backwardPlan )
  {
    fftw_execute( fft.backwardPlan );
    fac = 1.0 / vol;
    blas::Scal( ntot, fac, fft.inputComplexVec.Data(), 1);
  }

  if ( plan == fft.forwardPlan )
  {
    fftw_execute( fft.forwardPlan );
    fac = vol / double(ntot);
    blas::Scal( ntot, fac, fft.outputComplexVec.Data(), 1);
  }

  if ( plan == fft.backwardPlanR2C )
  {
    fftw_execute( fft.backwardPlanR2C );
    fac = 1.0 / vol;
    blas::Scal( ntot, fac, fft.inputVecR2C.Data(), 1);
  }

  if ( plan == fft.forwardPlanR2C )
  {
    fftw_execute( fft.forwardPlanR2C );
    fac = vol / double(ntot);
    blas::Scal( ntotR2C, fac, fft.outputVecR2C.Data(), 1);
  }
  return ;
} 

} // namespace isdf
