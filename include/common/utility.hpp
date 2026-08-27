#pragma once
#include  <stdlib.h>
#include  "domain.hpp"
#include  "environment.hpp"
#include  "tinyvec_impl.hpp"
#include  "numvec_impl.hpp"
#include  "nummat_impl.hpp"
#include  "numtns_impl.hpp"
#include <cstdlib>
#ifdef GPU
#include  "../cuda/cu_nummat_impl.hpp"
#include  "../cuda/cu_numvec_impl.hpp"
#include  "../cuda/cuda_utils.h"
#endif
namespace isdf{


inline Int IRound(Real a){ 
  Int b = 0;
  if(a>0) b = (a-Int(a)<0.5)?Int(a):(Int(a)+1);
  else b = (Int(a)-a<0.5)?Int(a):(Int(a)-1);
  return b; 
}


inline Int Size( std::stringstream& sstm ){
  Int length;
  sstm.seekg (0, std::ios::end);
  length = sstm.tellg();
  sstm.seekg (0, std::ios::beg);
  return length;
}




inline Int Print(std::ostream &os, const std::string name) {
  os << std::setiosflags(std::ios::left) << name << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char* name) {
  os << std::setiosflags(std::ios::left) << std::string(name) << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name, std::string val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name, const char* val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << std::string(val)
    << std::endl;
  return 0;
};



inline Int Print(std::ostream &os, const std::string name, Real val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char* name, Real val) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const std::string name, Real val, const std::string unit) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << unit 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name, Real val, const char *unit) {
  os << std::setiosflags(std::ios::left) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit) 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name1, Real val1, const std::string unit1,
    const std::string name2, Real val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name1
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val1
    << std::setw(LENGTH_VAR_UNIT) << unit1 
    << std::setw(LENGTH_VAR_NAME) << name2
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << unit2 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Real val1, const char *unit1,
    char *name2, Real val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val1
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name1, Int val1, const std::string unit1,
    const std::string name2, Real val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name1
    << std::setw(LENGTH_INT_DATA) << val1
    << std::setw(LENGTH_VAR_UNIT) << unit1 
    << std::setw(LENGTH_VAR_NAME) << name2
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << unit2 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Int val1, const char *unit1,
    char *name2, Real val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
    const char *name1, Int val1, 
    const char *name2, Real val2, 
    char *name3, Real val3) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1 
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
    << std::setw(LENGTH_VAR_NAME) << std::string(name3) 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val3
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, 
    const char *name1, Int val1, 
    const char *name2, Real val2, 
    const char *name3, Real val3,
    const char *name4, Real val4 ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1 
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
    << std::setw(LENGTH_VAR_NAME) << std::string(name3) 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val3
    << std::setw(LENGTH_VAR_NAME) << std::string(name4) 
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val4
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};



inline Int Print(std::ostream &os,
    const char *name1, Int val1, 
    const char *name2, Real val2,
    const char *name3, Real val3,
    const char *name4, Real val4,
    const char *name5, Real val5,
    const char *name6, Real val6,
    const char *name7, Real val7,
    const char *name8, Real val8) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
    << std::setw(LENGTH_VAR_NAME) << std::string(name3)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val3
    << std::setw(LENGTH_VAR_NAME) << std::string(name4)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val4
    << std::setw(LENGTH_VAR_NAME) << std::string(name5)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val5
    << std::setw(LENGTH_VAR_NAME) << std::string(name6)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val6
    << std::setw(LENGTH_VAR_NAME) << std::string(name7)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val7
    << std::setw(LENGTH_VAR_NAME) << std::string(name8)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val8
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, std::string name, Int val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char *name, Int val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setw(LENGTH_VAR_DATA) << val
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const std::string name, Int val, const std::string unit) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name
    << std::setw(LENGTH_VAR_DATA) << val
    << std::setw(LENGTH_VAR_UNIT) << unit 
    << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char* name, Int val, const std::string unit) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name)
    << std::setw(LENGTH_VAR_DATA) << val
    << std::setw(LENGTH_VAR_UNIT) << unit 
    << std::endl;
  return 0;
};



inline Int Print(std::ostream &os, const std::string name1, Int val1, const std::string unit1,
    const std::string name2, Int val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name1
    << std::setw(LENGTH_VAR_DATA) << val1
    << std::setw(LENGTH_VAR_UNIT) << unit1 
    << std::setw(LENGTH_VAR_NAME) << name2
    << std::setw(LENGTH_VAR_DATA) << val2
    << std::setw(LENGTH_VAR_UNIT) << unit2 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Int val1, const char *unit1,
    char *name2, Int val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_VAR_DATA) << val1
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setw(LENGTH_VAR_DATA) << val2
    << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name, bool val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << name;
  if( val == true )
    os << std::setw(LENGTH_VAR_NAME) << "true" << std::endl;
  else
    os << std::setw(LENGTH_VAR_NAME) << "false" << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char* name, bool val) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name);
  if( val == true )
    os << std::setw(LENGTH_VAR_NAME) << "true" << std::endl;
  else
    os << std::setw(LENGTH_VAR_NAME) << "false" << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, 
    const char *name1, Index3 val ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_VAR_DATA) << val[0]
    << std::setw(LENGTH_VAR_DATA) << val[1]
    << std::setw(LENGTH_VAR_DATA) << val[2]
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
    const char *name1, Point3 val ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[0]
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[1]
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[2]
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
    const char *name1, Int val1,
    const char *name2, Point3 val ) {
  os << std::setiosflags(std::ios::left)
    << std::setw(LENGTH_VAR_NAME) << std::string(name1)
    << std::setw(LENGTH_INT_DATA) << val1
    << std::setw(LENGTH_VAR_NAME) << std::string(name2)
    << std::setiosflags(std::ios::scientific)
    << std::setiosflags(std::ios::showpos)
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[0]
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[1]
    << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[2]
    << std::resetiosflags(std::ios::scientific)
    << std::resetiosflags(std::ios::showpos)
    << std::endl;
  return 0;
};


template <class F> inline std::ostream& operator<<( std::ostream& os, const std::vector<F>& vec)
{
  os<<vec.size()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<vec.size(); i++)     
    os<<" "<<vec[i];
  os<<std::endl;
  return os;
}

// NumVec
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumVec<F>& vec)
{
  os<<vec.m()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<vec.m(); i++)     
    os<<" "<<vec(i);
  os<<std::endl;
  return os;
}

template <class F> inline std::ostream& operator<<( std::ostream& os, const NumMat<F>& mat)
{
  os<<mat.m()<<" "<<mat.n()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<mat.m(); i++) {
    for(Int j=0; j<mat.n(); j++)
      os<<" "<<mat(i,j);
    os<<std::endl;
  }
  return os;
}

// NumTns
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumTns<F>& tns)
{
  os<<tns.m()<<" "<<tns.n()<<" "<<tns.p()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<tns.m(); i++) {
    for(Int j=0; j<tns.n(); j++) {
      for(Int k=0; k<tns.p(); k++) {
        os<<" "<<tns(i,j,k);
      }
      os<<std::endl;
    }
    os<<std::endl;
  }
  return os;
}

inline void GetTime(Real&  t){
  t = MPI_Wtime();
}


inline void SetRandomSeed(long int seed){
	  srand48(seed);
}


inline Real UniformRandom(){
  return (Real)drand48();
}


inline Real GaussianRandom(){
  Real a = UniformRandom(), b = UniformRandom();
  return std::sqrt(-2.0*std::log(a))*std::cos(2.0*PI*b);
}


inline void GaussianRandom( NumMat<Real>& M )
{
  Real *ptr = M.Data();
  for(Int i=0; i < M.m() * M.n(); i++) 
    *(ptr++) = GaussianRandom(); 
}
















void AlltoallForward( DblNumMat& A, DblNumMat& B, MPI_Comm comm );
void AlltoallForward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm );
void AlltoallBackward( DblNumMat& A, DblNumMat& B, MPI_Comm comm );
void AlltoallBackward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm );

#if defined(GPU)
void GPU_AlltoallBackward( cuDblNumMat& A, cuDblNumMat& B, MPI_Comm comm );
void GPU_AlltoallForward ( cuDblNumMat& A, cuDblNumMat& B, MPI_Comm comm );
#endif

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin);

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin, NumVec<Real>& amin);

void pdist2(NumMat<Real>& A, NumMat<Real>& B, NumMat<Real>& D);


void KMEAN(Int n, NumVec<Real>& weight, Int& rk, Real KmeansTolerance, 
    Int KmeansMaxIter, Real DFTolerance, const Domain &dm, Int* piv);


} // namespace isdf
