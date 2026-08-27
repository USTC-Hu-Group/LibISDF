#include "../../include/common/utility.hpp"
namespace isdf{


class SendReceiveMap {
   private:
    Int mpirank, mpisize, height, width;
    Int widthBlocksize;
    Int heightBlocksize;
    Int widthLocal;
    Int heightLocal;
    IntNumVec sendcounts;
    IntNumVec recvcounts;
    IntNumVec senddispls;
    IntNumVec recvdispls;

   public:
    SendReceiveMap(Int mpirank, Int mpisize, Int height, Int width) : mpirank(mpirank), mpisize(mpisize), height(height), width(width){
        this->widthBlocksize = width / mpisize;
        this->heightBlocksize = height / mpisize;
        this->heightLocal = this->heightBlocksize + (mpirank < (height % mpisize));
        this->widthLocal = this->widthBlocksize + (mpirank < (width % mpisize));
        this->senddispls.Resize(mpisize);
        this->sendcounts.Resize(mpisize);
        this->recvdispls.Resize(mpisize);
        this->recvcounts.Resize(mpisize);
        for (Int k = 0; k < mpisize; k++) {
            sendcounts[k] = heightBlocksize * widthLocal;
            if (k < (height % mpisize)) {
                sendcounts[k] = sendcounts[k] + widthLocal;
            }
        }
        for (Int k = 0; k < mpisize; k++) {
            recvcounts[k] = heightLocal * widthBlocksize;
            if (k < (width % mpisize)) {
                recvcounts[k] = recvcounts[k] + heightLocal;
            }
        }
        senddispls[0] = 0;
        recvdispls[0] = 0;
        for (Int k = 1; k < mpisize; k++) {
            this->senddispls[k] = senddispls[k - 1] + sendcounts[k - 1];
            this->recvdispls[k] = recvdispls[k - 1] + recvcounts[k - 1];
        }
    };

    Int sendk(Int i, Int j) {
        Int result;
        if ((height % mpisize) == 0) {
            result = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
        } else {
            if (i < ((height % mpisize) * (heightBlocksize + 1))) {
                result = senddispls[i / (heightBlocksize + 1)] + j * (heightBlocksize + 1) + i % (heightBlocksize + 1);
            } else {
                result = senddispls[(height % mpisize) + (i - (height % mpisize) * (heightBlocksize + 1)) / heightBlocksize] + j * heightBlocksize + (i - (height % mpisize) * (heightBlocksize + 1)) % heightBlocksize;
            }
        }
        return result;
    }
    Int recvk(Int i, Int j) {
        Int result;
        result = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
        return result;
    }
    IntNumVec getColsMapForward(){
        IntNumVec colsMapForward(this->width);
        for (int j = 0; j < width; j++) {
            int colRankLocal;     
            int colProcessorRank; 
            if (j / (widthBlocksize + 1) < width % mpisize) {
                colRankLocal = (j - (widthBlocksize + 1) * (j / (widthBlocksize + 1))) % (widthBlocksize + 1);
                colProcessorRank = j / (widthBlocksize + 1);
            } else {
                colRankLocal = (j - (widthBlocksize + 1) * (width % mpisize)) % widthBlocksize;
                colProcessorRank = (j - (widthBlocksize + 1) * (width % mpisize)) / widthBlocksize + width % mpisize;
            }
            colsMapForward[j] = colRankLocal * mpisize + colProcessorRank;
        }
        return colsMapForward;
    }
    IntNumVec getColsMapBackward(){
        IntNumVec  colsMapBackward(this->width);
        for (int j = 0; j < this->width; j++) {
            int colRankLocal = j / mpisize;     
            int colProcessorRank = j % mpisize; 
            if (colProcessorRank < (width % mpisize)) {
                colsMapBackward[j] = colProcessorRank * (width / mpisize + 1) + colRankLocal;
            } else {
                colsMapBackward[j] = (width / mpisize + 1) * (width % mpisize) + (colProcessorRank - (width % mpisize)) * (width / mpisize) + colRankLocal;
            }
        }
        return colsMapBackward;
    }
};

void AlltoallForward(DblNumMat& A, DblNumMat& B, MPI_Comm comm) {
    int mpirank, mpisize;
    MPI_Comm_rank(comm, &mpirank);
    MPI_Comm_size(comm, &mpisize);

    Int height = A.m();
    Int widthTemp = A.n();
    Int width = 0;
    MPI_Allreduce(&widthTemp, &width, 1, MPI_INT, MPI_SUM, comm);

    Int widthBlocksize = width / mpisize;
    Int heightBlocksize = height / mpisize;
    Int widthLocal = widthBlocksize;
    Int heightLocal = heightBlocksize;

    if (mpirank < (width % mpisize)) {
        widthLocal = widthBlocksize + 1;
    }

    if (mpirank < (height % mpisize)) {
        heightLocal = heightBlocksize + 1;
    }

    IntNumVec sendcounts(mpisize);
    IntNumVec recvcounts(mpisize);
    IntNumVec senddispls(mpisize);
    IntNumVec recvdispls(mpisize);

    for (Int k = 0; k < mpisize; k++) {
        sendcounts[k] = heightBlocksize * widthLocal;
        if (k < (height % mpisize)) {
            sendcounts[k] = sendcounts[k] + widthLocal;
        }
    }

    for (Int k = 0; k < mpisize; k++) {
        recvcounts[k] = heightLocal * widthBlocksize;
        if (k < (width % mpisize)) {
            recvcounts[k] = recvcounts[k] + heightLocal;
        }
    }

    senddispls[0] = 0;
    recvdispls[0] = 0;
    for (Int k = 1; k < mpisize; k++) {
        senddispls[k] = senddispls[k - 1] + sendcounts[k - 1];
        recvdispls[k] = recvdispls[k - 1] + recvcounts[k - 1];
    }
    SendReceiveMap sendRecvMap(mpirank, mpisize, height, width);
    A.Rearrange(mpisize);
    MPI_Alltoallv(A.Data(), &sendcounts[0], &senddispls[0], MPI_DOUBLE,
                  &B(0, 0), &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm);
    IntNumVec colsMap(B.n());
    colsMap = sendRecvMap.getColsMapForward();
    B.ExchangeCols(colsMap);
    A.RevertRearrange(mpisize);
    return;
}


void AlltoallBackward(DblNumMat& A, DblNumMat& B, MPI_Comm comm) {
    int mpirank, mpisize;
    MPI_Comm_rank(comm, &mpirank);
    MPI_Comm_size(comm, &mpisize);

    Int height = B.m();
    Int widthTemp = B.n();

    Int width = 0;
    MPI_Allreduce(&widthTemp, &width, 1, MPI_INT, MPI_SUM, comm);

    Int widthBlocksize = width / mpisize;
    Int heightBlocksize = height / mpisize;
    Int widthLocal = widthBlocksize;
    Int heightLocal = heightBlocksize;

    if (mpirank < (width % mpisize)) {
        widthLocal = widthBlocksize + 1;
    }

    if (mpirank < (height % mpisize)) {
        heightLocal = heightBlocksize + 1;
    }

    IntNumVec sendcounts(mpisize);
    IntNumVec recvcounts(mpisize);
    IntNumVec senddispls(mpisize);
    IntNumVec recvdispls(mpisize);

    for (Int k = 0; k < mpisize; k++) {
        sendcounts[k] = heightBlocksize * widthLocal;
        if (k < (height % mpisize)) {
            sendcounts[k] = sendcounts[k] + widthLocal;
        }
    }

    for (Int k = 0; k < mpisize; k++) {
        recvcounts[k] = heightLocal * widthBlocksize;
        if (k < (width % mpisize)) {
            recvcounts[k] = recvcounts[k] + heightLocal;
        }
    }

    senddispls[0] = 0;
    recvdispls[0] = 0;
    for (Int k = 1; k < mpisize; k++) {
        senddispls[k] = senddispls[k - 1] + sendcounts[k - 1];
        recvdispls[k] = recvdispls[k - 1] + recvcounts[k - 1];
    }
    IntNumVec colsMap(A.n());
    SendReceiveMap sendRecvMap(mpirank, mpisize, height, width);
    colsMap = sendRecvMap.getColsMapBackward();
    A.ExchangeCols(colsMap);

    MPI_Alltoallv(A.Data(), &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
                  B.Data(), &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm);

    B.RevertRearrange(mpisize);
    colsMap = sendRecvMap.getColsMapForward();
    A.ExchangeCols(colsMap);
    return;
} 





void AlltoallForward(CpxNumMat& A, CpxNumMat& B, MPI_Comm comm) {
    int mpirank, mpisize;
    MPI_Comm_rank(comm, &mpirank);
    MPI_Comm_size(comm, &mpisize);

    Int height = A.m();
    Int widthTemp = A.n();
    Int width = 0;
    MPI_Allreduce(&widthTemp, &width, 1, MPI_INT, MPI_SUM, comm);

    Int widthBlocksize = width / mpisize;
    Int heightBlocksize = height / mpisize;
    Int widthLocal = widthBlocksize;
    Int heightLocal = heightBlocksize;

    if (mpirank < (width % mpisize)) {
        widthLocal = widthBlocksize + 1;
    }

    if (mpirank < (height % mpisize)) {
        heightLocal = heightBlocksize + 1;
    }

    IntNumVec sendcounts(mpisize);
    IntNumVec recvcounts(mpisize);
    IntNumVec senddispls(mpisize);
    IntNumVec recvdispls(mpisize);

    for (Int k = 0; k < mpisize; k++) {
        sendcounts[k] = heightBlocksize * widthLocal;
        if (k < (height % mpisize)) {
            sendcounts[k] = sendcounts[k] + widthLocal;
        }
    }

    for (Int k = 0; k < mpisize; k++) {
        recvcounts[k] = heightLocal * widthBlocksize;
        if (k < (width % mpisize)) {
            recvcounts[k] = recvcounts[k] + heightLocal;
        }
    }

    senddispls[0] = 0;
    recvdispls[0] = 0;
    for (Int k = 1; k < mpisize; k++) {
        senddispls[k] = senddispls[k - 1] + sendcounts[k - 1];
        recvdispls[k] = recvdispls[k - 1] + recvcounts[k - 1];
    }
    SendReceiveMap sendRecvMap(mpirank, mpisize, height, width);
    A.Rearrange(mpisize);
    MPI_Alltoallv(A.Data(), &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX,
                  &B(0, 0), &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX, comm);
    IntNumVec colsMap(B.n());
    colsMap = sendRecvMap.getColsMapForward();
    B.ExchangeCols(colsMap);
    A.RevertRearrange(mpisize);
    return;
}


void AlltoallBackward(CpxNumMat& A, CpxNumMat& B, MPI_Comm comm) {
    int mpirank, mpisize;
    MPI_Comm_rank(comm, &mpirank);
    MPI_Comm_size(comm, &mpisize);

    Int height = B.m();
    Int widthTemp = B.n();

    Int width = 0;
    MPI_Allreduce(&widthTemp, &width, 1, MPI_INT, MPI_SUM, comm);

    Int widthBlocksize = width / mpisize;
    Int heightBlocksize = height / mpisize;
    Int widthLocal = widthBlocksize;
    Int heightLocal = heightBlocksize;

    if (mpirank < (width % mpisize)) {
        widthLocal = widthBlocksize + 1;
    }

    if (mpirank < (height % mpisize)) {
        heightLocal = heightBlocksize + 1;
    }

    IntNumVec sendcounts(mpisize);
    IntNumVec recvcounts(mpisize);
    IntNumVec senddispls(mpisize);
    IntNumVec recvdispls(mpisize);

    for (Int k = 0; k < mpisize; k++) {
        sendcounts[k] = heightBlocksize * widthLocal;
        if (k < (height % mpisize)) {
            sendcounts[k] = sendcounts[k] + widthLocal;
        }
    }

    for (Int k = 0; k < mpisize; k++) {
        recvcounts[k] = heightLocal * widthBlocksize;
        if (k < (width % mpisize)) {
            recvcounts[k] = recvcounts[k] + heightLocal;
        }
    }

    senddispls[0] = 0;
    recvdispls[0] = 0;
    for (Int k = 1; k < mpisize; k++) {
        senddispls[k] = senddispls[k - 1] + sendcounts[k - 1];
        recvdispls[k] = recvdispls[k - 1] + recvcounts[k - 1];
    }
    IntNumVec colsMap(A.n());
    SendReceiveMap sendRecvMap(mpirank, mpisize, height, width);
    colsMap = sendRecvMap.getColsMapBackward();
    A.ExchangeCols(colsMap);

    MPI_Alltoallv(A.Data(), &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX,
                  B.Data(), &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX, comm);

    B.RevertRearrange(mpisize);
    colsMap = sendRecvMap.getColsMapForward();
    A.ExchangeCols(colsMap);
    return;
} 









#ifdef GPU
void GPU_AlltoallForward( cuDblNumMat& cu_A, cuDblNumMat& cu_B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = cu_A.m();
  Int widthTemp = cu_A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }
  
  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }
  
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  cuIntNumMat  cu_sendk( height, widthLocal );
  cuIntNumMat  cu_recvk( heightLocal, width );
  cuIntNumVec  cu_senddispls(mpisize);
  cuIntNumVec  cu_recvdispls(mpisize);
  cuDblNumVec  cu_recvbuf(heightLocal*width);
  cuDblNumVec  cu_sendbuf(height*widthLocal); 

  cu_senddispls.CopyFrom( senddispls );
  cu_recvdispls.CopyFrom( recvdispls );
 
  cuda_cal_sendk_isdf( cu_sendk.Data(), cu_senddispls.Data(), widthLocal, height, heightBlocksize, mpisize );
  cuda_cal_recvk_isdf( cu_recvk.Data(), cu_recvdispls.Data(), width, heightLocal, mpisize ); 

  cuda_mapping_to_buf_isdf( cu_sendbuf.Data(), cu_A.Data(), cu_sendk.Data(), height*widthLocal);
  
#ifdef GPUDIRECT
  std::cout<<"gpu direct"<<std::endl;
  MPI_Alltoallv( cu_sendbuf.Data(), &sendcounts[0], &senddispls[0], MPI_DOUBLE,
      cu_recvbuf.Data(), &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );

#else
  cu_sendbuf.CopyTo( sendbuf );
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );
  cu_recvbuf.CopyFrom( recvbuf );
#endif
  cuda_mapping_from_buf_isdf(cu_B.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
 

  return ;
}        // -----  end of function GPU_AlltoallForward ----- 


void GPU_AlltoallBackward( cuDblNumMat& cu_A, cuDblNumMat& cu_B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = cu_B.m();
  Int widthTemp = cu_B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  cuIntNumMat  cu_sendk( height, widthLocal );
  cuIntNumMat  cu_recvk( heightLocal, width );
  cuIntNumVec  cu_senddispls(mpisize);
  cuIntNumVec  cu_recvdispls(mpisize);
  cuDblNumVec  cu_recvbuf(heightLocal*width);
  cuDblNumVec  cu_sendbuf(height*widthLocal); 

  cu_senddispls.CopyFrom( senddispls );
  cu_recvdispls.CopyFrom( recvdispls );
 
  cuda_cal_sendk_isdf( cu_sendk.Data(), cu_senddispls.Data(), widthLocal, height, heightBlocksize, mpisize );
  cuda_cal_recvk_isdf( cu_recvk.Data(), cu_recvdispls.Data(), width, heightLocal, mpisize ); 

  cuda_mapping_to_buf_isdf( cu_recvbuf.Data(), cu_A.Data(), cu_recvk.Data(), heightLocal*width);
  
#ifdef GPUDIRECT
  std::cout<<"gpu direct"<<std::endl;
  MPI_Alltoallv( cu_recvbuf.Data(), &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
      cu_sendbuf.Data(), &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );
#else
  cu_recvbuf.CopyTo( recvbuf );
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );
  cu_sendbuf.CopyFrom( sendbuf );
#endif
  cuda_mapping_from_buf_isdf(cu_B.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);
  
  return ;
}        // -----  end of function GPU_AlltoallBackward ----- 
#endif







void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin){
  int n = A.n_;
  int m = A.m_;
  if (Dim == 0){ 
    Imin.Resize(n);
    Int* Iptr = Imin.Data();
    for (int i = 0; i < n; i++){
      double* temp = A.VecData(i);
      Iptr[i] = std::distance(temp,std::min_element(temp,temp+m));
    }
  } else {
    Real* Aptr = A.Data();
    DblNumVec amin(m,1,Aptr);
    Imin.Resize(m);
    SetValue(Imin,0);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
      for (int i = 0; i < m; i++){
        for (int j = 1; j < n; j++){
        if (Aptr[i+j*m] < aptr[i]){
          
          aptr[i] = Aptr[i+j*m];
          Iptr[i] = j;
        }
      }
    }
  }
}

void findMin(NumMat<Real>& A, const int Dim, NumVec<Int>& Imin, NumVec<Real>& amin){
  int n = A.n_;
  int m = A.m_;
  if (Dim == 0){ 
    Imin.Resize(n);
    amin.Resize(n);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    Int d;
    for (int i = 0; i < n; i++){
      double* temp = A.VecData(i);
      d = std::distance(temp,std::min_element(temp,temp+m));
      Iptr[i] = d;
      aptr[i] = temp[d];
    }
  } else {
    Real* Aptr = A.Data();
    amin = DblNumVec(m,1,Aptr);
    Imin.Resize(m);
    SetValue(Imin,0);
    Int* Iptr = Imin.Data();
    Real* aptr = amin.Data();
    for (int i = 0; i < m; i++){
    for (int j = 1; j < n; j++){
  //    for (int i = 0; i < m; i++){
        if (Aptr[i+j*m] < aptr[i]){
          aptr[i] = Aptr[i+j*m];
          Iptr[i] = j;
        }
      }
    } 
  }
}

void pdist2(NumMat<Real>& A, NumMat<Real>& B, NumMat<Real>& D){
  D.Resize(A.m_, B.m_);
  Int Am = A.m_;
  Int Bm = B.m_;
  Real* Dptr = D.Data();
  Real* Aptr = A.Data();
  Real* Bptr = B.Data();
  
  Real d1,d2,d3;
  for (int j = 0; j < Bm;  j++) {
    for (int i = 0; i < Am; i++) {
      d1 = Aptr[i] - Bptr[j];
      d2 = Aptr[i+Am] - Bptr[j+Bm];
      d3 = Aptr[i+2*Am] - Bptr[j+2*Bm];
      Dptr[j*Am+i] = d1*d1 + d2*d2 + d3*d3;
    }
  }
}

void unique(NumVec<Int>& Index){
  Sort(Index);
  Int* Ipt = Index.Data();
  Int* it = std::unique(Ipt, Ipt + Index.m_);
  std::vector<Int> temp(Ipt, it); 
  delete[] Index.Data();
  Index.m_ = temp.size();
  Index.data_ = new Int[Index.m_];
  Ipt = Index.Data();
  for (int i = 0; i < Index.m_; i++){
    Ipt[i] = temp[i];
  }
}

void KMEAN(Int n, NumVec<Real>& weight, Int& rk, Real KmeansTolerance, 
    Int KmeansMaxIter, Real DFTolerance,  const Domain &dm, Int* piv)
{
  MPI_Barrier(dm.comm);
  int mpirank; MPI_Comm_rank(dm.comm, &mpirank);
  int mpisize; MPI_Comm_size(dm.comm, &mpisize);
  
  Real timeSta, timeEnd;
  Real timeSta2, timeEnd2;
  Real timeDist=0.0;
  Real timeMin=0.0;
  Real timeComm=0.0;
  Real time0 = 0.0;

  GetTime(timeSta);
  Real* wptr = weight.Data();
  int npt;
  std::vector<int> index(n);
  double maxW = 0.0;
  if(DFTolerance > 1e-16){
    maxW = findMax(weight);
    npt = 0;
    for (int i = 0; i < n;i++){
      if (wptr[i] > DFTolerance*maxW){
        index[npt] = i;
        npt++;
      }
    }
    index.resize(npt);
  } else {
    npt = n;
    for (int i = 0; i < n; i++){
      index[i] = i;
    }
  }

  if(npt < rk){
    int k0 = 0;
    int k1 = 0;
    for (int i = 0; i < npt; i++){
      if ( i == index[k0] ){
        piv[k0] = i;
        k0 = std::min(k0+1, rk-1);
      } else {
        piv[npt+k1] = i;
        k1++;
      }
    }
    std::random_shuffle(piv+npt,piv+n);
    return;
  } 
  int nptLocal = n/mpisize; // n
  int res = n%mpisize;//n
  if (mpirank < res){
    nptLocal++;
  }
  int indexSta = mpirank*nptLocal;
  if (mpirank >= res){
    indexSta += res;
  }
  std::vector<int> indexLocal(nptLocal);
  DblNumMat GridLocal(nptLocal,3);
  Real* glptr = GridLocal.Data();
  DblNumVec weightLocal(nptLocal);
  Real* wlptr = weightLocal.Data();

  int tmp;
  double len[3];
  double dx[3];
  int nG[3];
  for (int i = 0; i < 3; i++){
    len[i] = dm.length[i];
    nG[i] = dm.numGrid[i];
    dx[i] = len[i]/nG[i];
  }

  for (int i = 0; i < nptLocal; i++){
    tmp = index[indexSta+i];
    indexLocal[i] = tmp;
    wlptr[i] = wptr[tmp];
    glptr[i] = (tmp%nG[0])*dx[0];
    glptr[i+nptLocal] = (tmp%(nG[0]*nG[1])-glptr[i]/dx[0])/nG[0]*dx[1];
    glptr[i+2*nptLocal] = (tmp-glptr[i]/dx[0]-glptr[i+nptLocal]/dx[1]*nG[0])/(nG[0]*nG[1])*dx[2];
  
}
  DblNumMat C(rk,3);
  Real* Cptr = C.Data();
  std::vector<int> Cind = index;
  std::vector<int> Cinit;
  Cinit.reserve(rk);
//  std::random_shuffle(Cind.begin(), Cind.end());
  GetTime(timeEnd);
  isdfOFS << "After Setup: " << timeEnd-timeSta << "[s]" << std::endl;

  if (piv[0]!= piv[1]){
    isdfOFS << "Used previous initialization." << std::endl;
    for (int i = 0; i < rk; i++){
      if(wptr[piv[i]] > DFTolerance*maxW){
        Cinit.push_back(piv[i]);
      }
    }
    isdfOFS << "Reusable pivots: " << Cinit.size() << std::endl;
//    GetTime(timeEnd);
//    isdfOFS << "After load: " << timeEnd-timeSta << "[s]" << std::endl;
    int k = 0;
    while(Cinit.size() < rk && k < npt){
      bool flag = 1;
      int it = 0; 
      while (flag && it < Cinit.size()){
        if (Cinit[it] == Cind[k]){
          flag = 0;
        }
        it++;
      }
      if(flag){
        Cinit.push_back(Cind[k]);
      }
      k++;
    }
  } else {
    Cinit = Cind;
    Cinit.resize(rk);
  }
  GetTime(timeEnd);
  isdfOFS << "After Initialization: " << timeEnd-timeSta << "[s]" << std::endl;
  for (int i = 0; i < rk; i++){
    tmp = Cinit[i];
    Cptr[i] = (tmp%nG[0])*dx[0];
    Cptr[i+rk] = (tmp%(nG[0]*nG[1])-Cptr[i]/dx[0])/nG[0]*dx[1];
    Cptr[i+2*rk] = (tmp-Cptr[i]/dx[0]-Cptr[i+rk]/dx[1]*nG[0])/(nG[0]*nG[1])*dx[2];

  }

  int s = 0;
  int flag = n;
  int flagrecv = 0;
  IntNumVec label(nptLocal);
  Int* lbptr = label.Data();
  IntNumVec last(nptLocal);
  Int* laptr = last.Data();
  DblNumVec count(rk);
  Real* cptr = count.Data();
  DblNumMat DLocal(nptLocal, rk);
  DblNumMat Crecv(rk,3);
  Real* Crptr = Crecv.Data();
  DblNumVec countrecv(rk);
  Real* crptr = countrecv.Data();

  GetTime(timeSta2);
#if defined(GPU)
  //isdfOFS<<"1"<<std::endl;
  cuDblNumMat cu_DLocal(nptLocal, rk);
  cuDblNumMat cu_GridLocal(nptLocal,3);
  cuDblNumMat cu_C(rk,3);
  cuda_memcpy_CPU2GPU(cu_GridLocal.Data(), GridLocal.Data(), sizeof(double) * nptLocal*3);
  cuda_memcpy_CPU2GPU( cu_C.Data(), C.Data(), sizeof(double) * rk *3);
  pdist2_GPU(cu_GridLocal.Data(), cu_C.Data(), cu_DLocal.Data(), nptLocal,rk);
  //cuda_memcpy_GPU2CPU(DLocal.Data(), cu_DLocal.Data(), sizeof(double) * nptLocal*rk);
#else
  pdist2(GridLocal, C, DLocal);
#endif
  GetTime(timeEnd2);
  timeDist += (timeEnd2-timeSta2);
  
  GetTime(timeSta2);
  
#if defined(GPU)
  //isdfOFS<<"2"<<std::endl;
  cuIntNumVec cu_label(nptLocal);
  cuDblNumVec cu_amin(rk);
  findMin_GPU(cu_DLocal.Data(),cu_label.Data(),cu_amin.Data(),1,nptLocal,rk, false);
 // findMin_GPU(cu_DLocal.Data(),cu_label.Data(),cu_amin.Data(),1,cu_DLocal.m(),cu_DLocal.n(), false);
  //findMin_GPU(cu_DLocal.Data(),1,cu_label.Data(),cu_DLocal.m(),cu_DLocal.n());
  cuda_memcpy_GPU2CPU(label.Data(), cu_label.Data(), sizeof(int) * nptLocal);
  //printCpxV(cu_label,psiOFS);
#else
  findMin(DLocal, 1, label);
#endif
  GetTime(timeEnd2);
  timeMin+=(timeEnd2-timeSta2);
  lbptr = label.Data();
  double maxF = KmeansTolerance*n;
  while (flag > maxF && s < KmeansMaxIter){
    SetValue(count, 0.0);
    SetValue(C, 0.0);
    for (int i = 0; i < nptLocal; i++){
      tmp = lbptr[i];
      cptr[tmp] += wlptr[i];
      Cptr[tmp] += wlptr[i]*glptr[i];
      Cptr[tmp+rk] += wlptr[i]*glptr[i+nptLocal];
      Cptr[tmp+2*rk] += wlptr[i]*glptr[i+2*nptLocal];
    }
    MPI_Barrier(dm.comm);
    GetTime(timeSta2);
    MPI_Reduce(cptr, crptr, rk, MPI_DOUBLE, MPI_SUM, 0, dm.comm);
    MPI_Reduce(Cptr, Crptr, rk*3, MPI_DOUBLE, MPI_SUM, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    GetTime(timeSta2);
    if (mpirank == 0){
      tmp = rk;
      for (int i = 0; i < rk; i++){
        if(crptr[i] != 0.0){
          Crptr[i] = Crptr[i]/crptr[i];
          Crptr[i+tmp] = Crptr[i+tmp]/crptr[i];
          Crptr[i+2*tmp] = Crptr[i+2*tmp]/crptr[i];
        } else {
          rk--;
          Crptr[i] = Crptr[rk];
          Crptr[i+tmp] = Crptr[rk+tmp];
          Crptr[i+2*tmp] = Crptr[rk+2*tmp];
          crptr[i] = crptr[rk];
          i--;
        }
      }
      C.Resize(rk,3);
      Cptr = C.Data();
      for (int i = 0; i < rk; i++){
        Cptr[i] = Crptr[i];
        Cptr[i+rk] = Crptr[i+tmp];
        Cptr[i+2*rk] = Crptr[i+2*tmp];
      }
    }
    GetTime(timeEnd2);
    time0 += (timeEnd2-timeSta2);
    GetTime(timeSta2);
    MPI_Bcast(&rk, 1, MPI_INT, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    if (mpirank != 0){
      C.Resize(rk,3);
      Cptr= C.Data();
    }
    GetTime(timeSta2);
    MPI_Bcast(Cptr, rk*3, MPI_DOUBLE, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    count.Resize(rk);
    cptr = count.Data();
    GetTime(timeSta2);
#if defined(GPU)
    //isdfOFS<<"3"<<std::endl;
    //cuda_memcpy_CPU2GPU(cu_GridLocal.Data(), GridLocal.Data(), sizeof(double) * nptLocal*3);
    cuda_memcpy_CPU2GPU( cu_C.Data(), C.Data(), sizeof(double) * rk *3);
    pdist2_GPU(cu_GridLocal.Data(), cu_C.Data(), cu_DLocal.Data(), nptLocal,rk);
    cudaDeviceSynchronize();
//    cuda_memcpy_GPU2CPU(DLocal.Data(), cu_DLocal.Data(), sizeof(double) * nptLocal*rk);
#else
    pdist2(GridLocal, C, DLocal);
#endif
    GetTime(timeEnd2);
    timeDist += (timeEnd2-timeSta2);

    last = label;
    laptr = last.Data();
    GetTime(timeSta2);

#if defined(GPU)
    //isdfOFS<<"4"<<std::endl; 
    cudaDeviceSynchronize();   
    findMin_GPU(cu_DLocal.Data(),cu_label.Data(),cu_amin.Data(),1,nptLocal,rk, false);
    //cudaDeviceSynchronize();
    //findMin_GPU(cu_DLocal.Data(),cu_label.Data(),cu_amin.Data(),1,cu_DLocal.m(),cu_DLocal.n(), false);
    //findMin_GPU(cu_DLocal.Data(),1,cu_label.Data(),cu_DLocal.m(),cu_DLocal.n());
    cuda_memcpy_GPU2CPU(label.Data(), cu_label.Data(), sizeof(int) * nptLocal);
#else

    findMin(DLocal, 1, label);
#endif
    GetTime(timeEnd2);
    timeMin +=(timeEnd2-timeSta2);
    lbptr = label.Data();
    flag = 0;
    for (int i = 0; i < label.m_; i++){
      if(laptr[i]!=lbptr[i]){
        flag++;
      }
    }
    MPI_Barrier(dm.comm);
    GetTime(timeSta2);
    MPI_Reduce(&flag, &flagrecv, 1, MPI_INT, MPI_SUM, 0, dm.comm);
    MPI_Bcast(&flagrecv, 1, MPI_INT, 0, dm.comm);
    GetTime(timeEnd2);
    timeComm += (timeEnd2-timeSta2);

    flag = flagrecv;
    isdfOFS<< flag << " ";
    s++;
  }
  isdfOFS << std::endl << "Converged in " << s << " iterations." << std::endl;
  GetTime(timeEnd);
  isdfOFS << "After iteration: " << timeEnd-timeSta << "[s]" << std::endl;
  IntNumVec Imin(rk);
  Int* imptr = Imin.Data();
  DblNumVec amin(rk);
  GetTime(timeSta2);
#if defined(GPU)
  //isdfOFS<<"5"<<std::endl;
  findMin_GPU(cu_DLocal.Data(),cu_label.Data(),cu_amin.Data(),0,nptLocal,rk, true);
   // findMin_GPU(cu_DLocal.Data(),cu_label.Data(),cu_amin.Data(),0,cu_DLocal.m(),cu_DLocal.n(), true);
    //findMin_GPU(cu_DLocal.Data(),0,cu_label.Data(),cu_DLocal.m(),cu_DLocal.n());
    cuda_memcpy_GPU2CPU(Imin.Data(), cu_label.Data(), sizeof(int) * rk);
    cuda_memcpy_GPU2CPU(amin.Data(), cu_amin.Data(), sizeof(double) * rk);
#else
  findMin(DLocal, 0, Imin, amin);
  //printCpxM(DLocal,psiOFS);
#endif
  GetTime(timeEnd2);
  timeMin += (timeEnd2-timeSta2);
  for (int i = 0; i < rk; i++){
    imptr[i] = indexLocal[imptr[i]];
  }
struct ValIdx {
    double val;
    int    idx;
};

std::vector<ValIdx> local_min(rk), global_min(rk);
for (int i = 0; i < rk; i++) {
    local_min[i].val = amin.Data()[i];
    local_min[i].idx = imptr[i];
}

GetTime(timeSta2);
MPI_Allreduce(local_min.data(), global_min.data(), rk,
              MPI_DOUBLE_INT, MPI_MINLOC, dm.comm);
GetTime(timeEnd2);
timeComm += (timeEnd2 - timeSta2);

IntNumVec pivTemp(rk);
Int* pvptr = pivTemp.Data();
for (int i = 0; i < rk; i++) {
    pvptr[i] = global_min[i].idx;
}


  unique(pivTemp);
  pvptr = pivTemp.Data();
  rk = pivTemp.m_;
  int k0 = 0;
  int k1 = 0;
  for (int i = 0; i < n; i++){
    if(i == pvptr[k0]){
      piv[k0] = i;
      k0 = std::min(k0+1, rk-1);
    } else {
      piv[rk+k1] = i;
      k1++;
    }
  }
  isdfOFS << "Dist time: " << timeDist << "[s]" << std::endl;
  isdfOFS << "Min time: " << timeMin << "[s]" << std::endl;
  isdfOFS << "Comm time: " << timeComm << "[s]" << std::endl;
  isdfOFS << "core0 time: " << time0 << "[s]" << std::endl;
}




}  // namespace isdf
