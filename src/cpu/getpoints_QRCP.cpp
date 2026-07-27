#include "../../include/isdf.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
//using namespace isdf;
using namespace isdf::scalapack;
namespace isdf{
#if !defined(GPU)
void isdffunc::getpoints_QRCP(DblNumMat &psiCol, Domain &domain_, const Int &nv_, const Int &nv1, const Int &nc1, const Int &nv2, const Int &nc2, Int &ntotLocal, Int rk, IntNumVec &pivQR_)
{
  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  Index3& numGrid = domain_.numGrid;
  Int ntot     = domain_.NumGridTotal();
  Int numStateTotal=0;
  Int numStateLocal = psiCol.n();
  Int Ng = ntot;
  Int numMu_ = rk;
  MPI_Allreduce(&numStateLocal, &numStateTotal, 1, MPI_INT, MPI_SUM, domain_.comm);
  Int Ne = numStateTotal;
  Int desc_NgNe1DCol[9];
  Int desc_NgNe1DRow[9];
  Int contxt0, contxt1, contxt11;
  Int nprow0, npcol0, myrow0, mycol0, info0;
  Int nprow1, npcol1, myrow1, mycol1, info1;
  Int nprow11, npcol11, myrow11, mycol11, info11;
  Int nprow2, npcol2, myrow2, mycol2, info2;

  Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol;
  Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
  nprow1 = 1;
  npcol1 = mpisize;
  Int I_ONE = 1, I_ZERO = 0;

  Cblacs_get(0, 0, &contxt1);
  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);

  if(contxt1 >= 0){
    nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNe1DCol = SCALAPACK(numroc)(&numStateTotal, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
  }    
  isdfOFS<<"numStateLocal "<<numStateLocal<<std::endl;

  SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &numStateTotal, &Ng, &I_ONE, &I_ZERO, 
      &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);



  nprow11 = mpisize;
  npcol11 = 1;

  Cblacs_get(0, 0, &contxt11);
  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);
  Int BlockSizeScaLAPACK = 1;
  Int BlockSizeScaLAPACKTemp = BlockSizeScaLAPACK; 
  //desc_NgNe1DRow
  if(contxt11 >= 0){
    nrowsNgNe1DRow = SCALAPACK(numroc)(&ntot, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
    ncolsNgNe1DRow = SCALAPACK(numroc)(&numStateTotal, &numStateTotal, &mycol11, &I_ZERO, &npcol11);
    lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
  }
  isdfOFS<<"ncolsNgNe1DRow"<<ncolsNgNe1DRow<<std::endl;
  isdfOFS<<"nrowsNgNe1DRow"<<nrowsNgNe1DRow<<std::endl;
  isdfOFS<<"ncolsNgNe1DCol"<<ncolsNgNe1DCol<<std::endl;
  isdfOFS<<"nrowsNgNe1DCol"<<nrowsNgNe1DCol<<std::endl;
  isdfOFS<<"numStateTotal "<<numStateTotal<<std::endl;
  SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &numStateTotal, &BlockSizeScaLAPACKTemp, &numStateTotal, &I_ZERO,&I_ZERO, &contxt11, &lldNgNe1DRow, &info11);
  DblNumMat psiRow( ntotLocal, numStateTotal );
  SetValue( psiRow, 0.0 );

  SCALAPACK(pdgemr2d)(&Ng, &numStateTotal, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol,psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );


  Real numGaussianRandomFac = 1.5 ;
  Int numPre = IRound(std::sqrt(numMu_*numGaussianRandomFac));
  if( numPre > numStateTotal ){
    ErrorHandling("numMu is too large for interpolative separable density fitting!");
  }

  isdfOFS << "ntot          = " << ntot << std::endl;
  isdfOFS << "numMu         = " << numMu_ << std::endl;
  isdfOFS << "numPre        = " << numPre << std::endl;
  isdfOFS << "numPre*numPre = " << numPre * numPre << std::endl;

  Int numStateBlocksize = numStateTotal / mpisize;

  Int numMuBlocksize = numMu_ / mpisize;

  Int desc_NeNe0D[9];


  // 0D MPI
  nprow0 = 1;
  npcol0 = mpisize;

  Cblacs_get(0, 0, &contxt0);
  Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
  Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

  SCALAPACK(descinit)(desc_NeNe0D, &Ne, &Ne, &Ne, &Ne, &I_ZERO, &I_ZERO, &contxt0, &Ne, &info0);
  nprow11 = mpisize;
  npcol11 = 1;



  
  // 2D MPI
  
  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
    nprow2 = i; npcol2 = mpisize / nprow2;
    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
  }

//  Cblacs_get(0, 0, &contxt2);
  //Cblacs_gridinit(&contxt2, "C", nprow2, npcol2);

//  IntNumVec pmap2(mpisize);
//  for ( Int i = 0; i < mpisize; i++ ){
 //   pmap2[i] = i;
 // }
 // Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);

 // Int mb2 = BlockSizeScaLAPACK;
 // Int nb2 = BlockSizeScaLAPACK;

  // Computing the indices is optional
 ntotLocal = nrowsNgNe1DRow; 

  Int ntotLocalMG, ntotMG;

 {

    GetTime( timeSta );

    DblNumMat localphiGRow( ntotLocal, numPre );
    SetValue( localphiGRow, 0.0 );

    DblNumMat localpsiGRow( ntotLocal, numPre );
    SetValue( localpsiGRow, 0.0 );

    DblNumMat G1( nc1 + nv1, numPre);
    SetValue( G1, 0.0 );
    DblNumMat G2( nc2 + nv2, numPre);
    SetValue( G2, 0.0 );

    
   if ( mpirank == 0) {
      bool useFile = false; 
      if (useFile) {
        isdfOFS << "Using file input mode for G1 and G2 matrices" << std::endl;
        
        
        std::ifstream file_G1("G1", std::ios::binary);
        if (file_G1.is_open()) {
          file_G1.read(reinterpret_cast<char*>(G1.Data()), (nc1 + nv1) * numPre * sizeof(Real));
          file_G1.close();
          isdfOFS << "G1 matrix loaded from file G1" << std::endl;
        } else {
          isdfOFS << "Error: Cannot open file G1, falling back to random generation" << std::endl;
          GaussianRandom(G1);
        }
        
      
        std::ifstream file_G2("G2", std::ios::binary);
        if (file_G2.is_open()) {
          file_G2.read(reinterpret_cast<char*>(G2.Data()), (nc2 + nv2) * numPre * sizeof(Real));
          file_G2.close();
          isdfOFS << "G2 matrix loaded from file G2" << std::endl;
        } else {
          isdfOFS << "Error: Cannot open file G2, falling back to random generation" << std::endl;
          GaussianRandom(G2);
        }
      } else {
        isdfOFS << "Using random generation mode for G1 and G2 matrices" << std::endl;
        GaussianRandom(G1);
        GaussianRandom(G2);
      }


      lapack::Orth( nc1 + nv1, numPre, G1.Data(), nc1 + nv1 );
      lapack::Orth( nc2 + nv2, numPre, G2.Data(), nc2 + nv2 );
    }
   
    GetTime( timeSta1 );

    MPI_Bcast(G1.Data(), (nv1 + nc1) * numPre, MPI_DOUBLE, 0, domain_.comm);
    MPI_Bcast(G2.Data(), (nv2 + nc2) * numPre, MPI_DOUBLE, 0, domain_.comm);

    GetTime( timeEnd1 );

    GetTime( timeSta1 );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, (nc1 + nv1), 1.0, 
        psiRow.Data()+ (nv_-nv1) * ntotLocal, ntotLocal, G1.Data(), (nc1 + nv1), 0.0,
        localpsiGRow.Data(), ntotLocal );

    blas::Gemm( 'N', 'N', ntotLocal, numPre, (nc2 + nv2), 1.0, 
        psiRow.Data()+ (nv_-nv2) * ntotLocal, ntotLocal, G2.Data(), (nc2 + nv2), 0.0,
        localphiGRow.Data(), ntotLocal );


    GetTime( timeEnd1 );

    int m_MGTemp = numPre*numPre;
    int n_MGTemp = ntot;

    DblNumMat MGCol( m_MGTemp, ntotLocal );

    DblNumVec MGNorm(ntot);
    for( Int k = 0; k < ntot; k++ ){
      MGNorm(k) = 0;
    }
    DblNumVec MGNormLocal(ntotLocal);
    for( Int k = 0; k < ntotLocal; k++ ){
      MGNormLocal(k) = 0;
    }

    GetTime( timeSta1 );

    for( Int j = 0; j < numPre; j++ ){
      for( Int i = 0; i < numPre; i++ ){
        for( Int ir = 0; ir < ntotLocal; ir++ ){
          MGCol(i+j*numPre,ir) = localphiGRow(ir,i) * localpsiGRow(ir,j);
          MGNormLocal(ir) += MGCol(i+j*numPre,ir) * MGCol(i+j*numPre,ir);   
        }
      }
    }

    GetTime( timeEnd1 );



    IntNumVec MGIdx(ntot);

    {
      Int ncols0D, nrows0D, lld0D; 
      Int ncols1D, nrows1D, lld1D; 

      Int desc_0D[9];
      Int desc_1D[9];

//      Cblacs_get(0, 0, &contxt0);
//      Cblacs_gridinit(&contxt0, "C", nprow0, npcol0);
//      Cblacs_gridinfo(contxt0, &nprow0, &npcol0, &myrow0, &mycol0);

      SCALAPACK(descinit)(desc_0D, &Ng, &I_ONE, &Ng, &I_ONE, &I_ZERO, &I_ZERO, &contxt0, &Ng, &info0);

      if(contxt11 >= 0){
        nrows1D = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACKTemp, &myrow11, &I_ZERO, &nprow11);
        ncols1D = SCALAPACK(numroc)(&I_ONE, &I_ONE, &mycol11, &I_ZERO, &npcol11);
        lld1D = std::max( nrows1D, 1 );
      }    

      SCALAPACK(descinit)(desc_1D, &Ng, &I_ONE, &BlockSizeScaLAPACKTemp, &I_ONE, &I_ZERO, 
          &I_ZERO, &contxt11, &lld1D, &info11);


      SCALAPACK(pdgemr2d)(&Ng, &I_ONE, MGNormLocal.Data(), &I_ONE, &I_ONE, desc_1D, 
          MGNorm.Data(), &I_ONE, &I_ONE, desc_0D, &contxt11 );

      MPI_Bcast( MGNorm.Data(), ntot, MPI_DOUBLE, 0, domain_.comm );


      double MGNormMax = *(std::max_element( MGNorm.Data(), MGNorm.Data() + ntot ) );
      double MGNormMin = *(std::min_element( MGNorm.Data(), MGNorm.Data() + ntot ) );

      ntotMG = 0;
      SetValue( MGIdx, 0 );
      Real relTol = 1e-5; 
      Real hybridDFTolerance = MGNormMax * relTol;
      for( Int k = 0; k < ntot; k++ ){
        if(MGNorm(k) > hybridDFTolerance){
          MGIdx(ntotMG) = k; 
          ntotMG = ntotMG + 1;
        }
      }

      isdfOFS << "The col size for MG: " << " ntotMG = " << ntotMG << " ntotMG/ntot = " << Real(ntotMG)/Real(ntot) << std::endl << std::endl;
      isdfOFS << "The norm range for MG: " <<  " MGNormMax = " << MGNormMax << " MGNormMin = " << MGNormMin << std::endl << std::endl;

    }

    Int ncols1DCol, nrows1DCol, lld1DCol; 
    Int ncols1DRow, nrows1DRow, lld1DRow; 

    Int desc_1DCol[9];
    Int desc_1DRow[9];

    if(contxt1 >= 0){
      nrows1DCol = SCALAPACK(numroc)(&m_MGTemp, &m_MGTemp, &myrow1, &I_ZERO, &nprow1);
      ncols1DCol = SCALAPACK(numroc)(&n_MGTemp, &BlockSizeScaLAPACKTemp, &mycol1, &I_ZERO, &npcol1);
      lld1DCol = std::max( nrows1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_1DCol, &m_MGTemp, &n_MGTemp, &m_MGTemp, &BlockSizeScaLAPACKTemp, &I_ZERO, 
        &I_ZERO, &contxt1, &lld1DCol, &info1);

    if(contxt11 >= 0){
      nrows1DRow = SCALAPACK(numroc)(&m_MGTemp, &I_ONE, &myrow11, &I_ZERO, &nprow11);
      ncols1DRow = SCALAPACK(numroc)(&n_MGTemp, &n_MGTemp, &mycol11, &I_ZERO, &npcol11);
      lld1DRow = std::max( nrows1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_1DRow, &m_MGTemp, &n_MGTemp, &I_ONE, &n_MGTemp, &I_ZERO, 
        &I_ZERO, &contxt11, &lld1DRow, &info11);

    DblNumMat MGRow( nrows1DRow, ntot );
    SetValue( MGRow, 0.0 );


    GetTime( timeSta1 );

    SCALAPACK(pdgemr2d)(&m_MGTemp, &n_MGTemp, MGCol.Data(), &I_ONE, &I_ONE, desc_1DCol, 
        MGRow.Data(), &I_ONE, &I_ONE, desc_1DRow, &contxt1 );

    GetTime( timeEnd1 );

    DblNumMat MG( nrows1DRow, ntotMG );

    for( Int i = 0; i < nrows1DRow; i++ ){
      for( Int j = 0; j < ntotMG; j++ ){
        MG(i,j) = MGRow(i,MGIdx(j));
      }
    }

    DblNumVec tau(ntotMG);


    for( Int k = 0; k < ntotMG; k++ ){
      tau[k] = 0.0;
    }

    Real timeQRCPSta, timeQRCPEnd;
    GetTime( timeQRCPSta );

    if(1){ //ScaLAPACL QRCP 2D

      Int contxt1D, contxt2D;
      Int nprow1D, npcol1D, myrow1D, mycol1D, info1D;
      Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

      Int ncols1D, nrows1D, lld1D; 
      Int ncols2D, nrows2D, lld2D; 

      Int desc_MG1D[9];
      Int desc_MG2D[9];

      Int m_MG = numPre*numPre;
      Int n_MG = ntotMG;

      Int mb_MG1D = 1;
      Int nb_MG1D = ntotMG;

      nprow1D = mpisize;
      npcol1D = 1;

      Cblacs_get(0, 0, &contxt1D);
      Cblacs_gridinit(&contxt1D, "C", nprow1D, npcol1D);
      Cblacs_gridinfo(contxt1D, &nprow1D, &npcol1D, &myrow1D, &mycol1D);

      nrows1D = SCALAPACK(numroc)(&m_MG, &mb_MG1D, &myrow1D, &I_ZERO, &nprow1D);
      ncols1D = SCALAPACK(numroc)(&n_MG, &nb_MG1D, &mycol1D, &I_ZERO, &npcol1D);

      lld1D = std::max( nrows1D, 1 );

      SCALAPACK(descinit)(desc_MG1D, &m_MG, &n_MG, &mb_MG1D, &nb_MG1D, &I_ZERO, 
          &I_ZERO, &contxt1D, &lld1D, &info1D);

      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(n_MG/m_MG))))); 
          i <= mpisize; i++){
        npcol2D = i; nprow2D = mpisize / npcol2D;
        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
      }


      Cblacs_get(0, 0, &contxt2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
  
    Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);


      Int m_MG2DBlocksize = BlockSizeScaLAPACK;
      Int n_MG2DBlocksize = BlockSizeScaLAPACK;

      Int m_MG2Local, n_MG2Local;

      MPI_Comm colComm = MPI_COMM_NULL;

  //    Int mpirankRow, mpisizeRow;
      Int mpirankCol, mpisizeCol;
      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );
      MPI_Comm_rank(colComm, &mpirankCol);
      MPI_Comm_size(colComm, &mpisizeCol);



      if(contxt2D >= 0){
        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
        nrows2D = SCALAPACK(numroc)(&m_MG, &m_MG2DBlocksize, &myrow2D, &I_ZERO, &nprow2D);
        ncols2D = SCALAPACK(numroc)(&n_MG, &n_MG2DBlocksize, &mycol2D, &I_ZERO, &npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      SCALAPACK(descinit)(desc_MG2D, &m_MG, &n_MG, &m_MG2DBlocksize, &n_MG2DBlocksize, &I_ZERO, 
          &I_ZERO, &contxt2D, &lld2D, &info2D);

      m_MG2Local = nrows2D;
      n_MG2Local = ncols2D;

      IntNumVec pivQRTmp1(ntotMG), pivQRTmp2(ntotMG), pivQRLocal(ntotMG);
      if( m_MG > ntot ){
        std::ostringstream msg;
        msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
  

      DblNumMat&  MG1D = MG;
      DblNumMat  MG2D (m_MG2Local, n_MG2Local);

      SCALAPACK(pdgemr2d)(&m_MG, &n_MG, MG1D.Data(), &I_ONE, &I_ONE, desc_MG1D, 
          MG2D.Data(), &I_ONE, &I_ONE, desc_MG2D, &contxt1D );
      SetValue( pivQRTmp1, 0 );

      if(contxt2D >= 0){
        Real timeQRCP1, timeQRCP2;
        GetTime( timeQRCP1 );
        scalapack::QRCPF( m_MG, n_MG, MG2D.Data(), desc_MG2D, pivQRTmp1.Data(), tau.Data() );
        GetTime( timeQRCP2 );
      }

      // Combine the local pivQRTmp to global pivQR_
      SetValue( pivQRLocal, 0 );
      for( Int j = 0; j < n_MG2Local; j++ ){
        pivQRLocal[ (j / n_MG2DBlocksize) * n_MG2DBlocksize * npcol2D + mycol2D * n_MG2DBlocksize + j % n_MG2DBlocksize] = pivQRTmp1[j];
      }

      SetValue( pivQRTmp2, 0 );
      MPI_Allreduce( pivQRLocal.Data(), pivQRTmp2.Data(), 
          ntotMG, MPI_INT, MPI_SUM, colComm );

      SetValue( pivQR_, 0 );
      for( Int j = 0; j < ntotMG; j++ ){
        pivQR_(j) = MGIdx(pivQRTmp2(j));
      }

      //if( rowComm != MPI_COMM_NULL ) MPI_Comm_free( & rowComm );
      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );

      if(contxt2D >= 0) {
        Cblacs_gridexit( contxt2D );
      }

      if(contxt1D >= 0) {
        Cblacs_gridexit( contxt1D );
      }

    } // if(1) ScaLAPACL QRCP

    GetTime( timeQRCPEnd );

    isdfOFS << "Time for QRCP alone is " <<
      timeQRCPEnd - timeQRCPSta << " [s]" << std::endl << std::endl;

    GetTime( timeEnd );
    isdfOFS << "Time for ISDF with QRCP is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
  }
  if(contxt0 >= 0) {
    Cblacs_gridexit( contxt0 );
  }

  if(contxt1 >= 0) {
    Cblacs_gridexit( contxt1 );
  }

  if(contxt11 >= 0) {
    Cblacs_gridexit( contxt11 );
  }

  //if(contxt2 >= 0) {
  //  Cblacs_gridexit( contxt2 );
  //}

  MPI_Barrier(domain_.comm);

  return ;
}       

#endif





}
