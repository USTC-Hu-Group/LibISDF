#pragma once
// STL libraries
#include <iostream> 
#include <iomanip> 
#include <fstream>
#include <sstream>
#include <unistd.h>

#include <cfloat>
#include <complex>
#include <string>
#include <cstring>

#include <set>
#include <map>
#include <stack>
#include <vector>

#include <algorithm>
#include <cmath>

#include <cassert>
#include <stdexcept>

// FFTW libraries
#include <fftw3.h>
#include <fftw3-mpi.h>
//COSMA
#include "./cosma_helper.hpp"
#include "mpi.h"



// The verbose level of debugging information
#ifdef  DEBUG
#define _DEBUGlevel_ DEBUG
#endif

#ifdef RELEASE
#define _RELEASE_
#define _DEBUGlevel -1
#endif





/***********************************************************************
 *  Data types and constants
 **********************************************************************/

namespace isdf{

// Basic data types

#define BLAS(name)      name##_
#define LAPACK(name)    name##_
#define SCALAPACK(name) name##_
#define F2C(name)       name##_

typedef    int                   Int;
typedef    double                Real;
typedef    std::complex<double>  Complex; 

// IO
extern  std::ofstream  isdfOFS;
extern  std::ofstream  psiOFS;


const Int I_ZERO = 0;
const Int I_ONE = 1;
const Real D_ZERO = 0.0;
const Real D_ONE  = 1.0;
const Complex Z_ZERO = Complex(0.0, 0.0);
const Complex Z_ONE  = Complex(1.0, 0.0);
const char UPPER = 'U';
const char LOWER = 'L';


const Int DIM = 3;                            // Always in 3D
const Real au2K = 315774.67;
const Real au2ev = 27.21138624598803130466;
const Real au2ang = 0.529177210903;
const Real amu2au =  1822.88848621731317242395;
const Real SPEED_OF_LIGHT = 137.0359895;
const Real PI = 3.14159265358979323846;
const Real au2as = 24.188843;
const Real au2fs = 0.024188843;
const Real Plank_SI =  6.62607015e-34;   //J s
const Real Boltzmann_SI = 1.380649e-23;  // J K^-1
const Real Electroncharge_SI = 1.602176634e-19;  //C
const Real ElectronVolt_SI = 1.602176634e-19;  //J
const Real Electronmass_SI =  9.1093837015e-31;  //Kg
const Real Hartree_SI = 4.3597447222071e-18; //J
const Real Ryberg_SI = Hartree_SI /2 ; // J
const Real Bohr_SI = 0.529177210903e-10; // m
const Real Atommu_SI = 1.66053906660e-27; // Kg
const Real Lightspeed_SI = 2.99792458e8;  //m s^-1
namespace DensityComponent{
enum {RHO, MAGX, MAGY, MAGZ};  
}
const int LENGTH_VAR_NAME = 8;
const int LENGTH_DBL_DATA = 16;
const int LENGTH_INT_DATA = 6;
const int LENGTH_VAR_UNIT = 6;
const int LENGTH_DBL_PREC = 12;
const int LENGTH_FULL_PREC = 16;
const int LENGTH_VAR_DATA = 20;

} // namespace isdf

/***********************************************************************
 *  Error handling
 **********************************************************************/

namespace isdf{


void ErrorHandling( const char * msg );

inline void ErrorHandling( const std::string& msg ){ ErrorHandling( msg.c_str() ); }

inline void ErrorHandling( const std::ostringstream& msg ) {ErrorHandling( msg.str().c_str() );}

struct NullStream : std::ostream
{            
  struct NullStreamBuffer : std::streambuf
  {
    Int overflow( Int c ) { return traits_type::not_eof(c); }
  } nullStreamBuffer_;

  NullStream() 
    : std::ios(&nullStreamBuffer_), std::ostream(&nullStreamBuffer_)
    { }
};  
} // namespace isdf


