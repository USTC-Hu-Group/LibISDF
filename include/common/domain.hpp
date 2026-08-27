#pragma once
#include  "environment.hpp"
#include  "tinyvec_impl.hpp"
#include  "numvec_impl.hpp"

namespace isdf{

struct Domain
{
  Point3       length;                         
  Point3       posStart;                       
  Index3       numGrid;                        
  Index3       numGridFine;                   
  MPI_Comm     comm;                           
  MPI_Comm     rowComm;
  MPI_Comm     colComm;

  Domain()
  { 
    length        = Point3( 0.0, 0.0, 0.0 );
    posStart      = Point3( 0.0, 0.0, 0.0 );
    numGrid       = Index3( 0, 0, 0 );
    numGridFine   = Index3( 0, 0, 0 );

    comm    = MPI_COMM_WORLD; 
    rowComm = MPI_COMM_WORLD;
    colComm = MPI_COMM_WORLD;
  }

  ~Domain(){}

  Real Volume() const { return length[0] * length[1] * length[2]; }
  Int  NumGridTotal() const { return numGrid[0] * numGrid[1] * numGrid[2]; }
  Int  NumGridTotalFine() const { return numGridFine[0] * numGridFine[1] * numGridFine[2]; }

};


} // namespace isdf


