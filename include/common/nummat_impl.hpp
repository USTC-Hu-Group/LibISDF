#pragma once
#include "nummat_decl.hpp"

namespace  isdf{

template <class F> inline NumMat<F>::NumMat(Int m, Int n): m_(m), n_(n), owndata_(true) {
  if(m_>0 && n_>0) { data_ = new F[(size_t)m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
}

template <class F> inline NumMat<F>::NumMat(Int m, Int n, bool owndata, F* data): m_(m), n_(n), owndata_(owndata) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = new F[(size_t)m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
    if(m_>0 && n_>0) { for(size_t i=0; i<(size_t) m_*n_; i++) data_[i] = data[i]; }
  } else {
    data_ = data;
  }
}

template <class F> inline NumMat<F>::NumMat(const NumMat& C): m_(C.m_), n_(C.n_), owndata_(C.owndata_) {
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = new F[(size_t)m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
    if(m_>0 && n_>0) { for(size_t i=0; i<(size_t)m_*n_; i++) data_[i] = C.data_[i]; }
  } else {
    data_ = C.data_;
  }
}
template <class F>
inline void NumMat<F>::FreeData() {
        if (data_ != NULL && owndata_) {
        delete[] data_;    
        data_ = NULL;      
        m_ = 0;            
        n_ = 0;
        owndata_ = false; 
        }
}
template <class F> inline NumMat<F>::~NumMat() {
  if(owndata_) {
    if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
  }
}

template <class F> inline NumMat<F>& NumMat<F>::operator=(const NumMat& C) {
  if(C.data_ == data_) return *this;
  if(owndata_) {
    if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
  }
  m_ = C.m_; n_=C.n_; owndata_=C.owndata_;
  if(owndata_) {
    if(m_>0 && n_>0) { data_ = new F[(size_t) m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
    if(m_>0 && n_>0) { for(size_t i=0; i<(size_t)m_*n_; i++) data_[i] = C.data_[i]; }
  } else {
    data_ = C.data_;
  }
  return *this;
}

template <class F> inline void NumMat<F>::Resize(Int m, Int n)  {
  if( owndata_ == false ){
    ErrorHandling("Matrix being resized must own data.");
  }
  if(m_!=m || n_!=n) {
    if(m_>0 && n_>0) { delete[] data_; data_ = NULL; }
    m_ = m; n_ = n;
    if(m_>0 && n_>0) { data_ = new F[(size_t)m_*n_]; if( data_ == NULL ) ErrorHandling("Cannot allocate memory."); } else data_=NULL;
  }
}


template <class F>
inline void NumMat<F>::Reset() {
         if (data_ != NULL && owndata_) {
              delete[] data_;}
              data_ = NULL;
              m_ = 0;
              n_ = 0;
              owndata_ = true;
}




template <class F> 
inline const F& NumMat<F>::operator()(Int i, Int j) const  { 
#if ( _DEBUGlevel_ >= 1 )
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return data_[i+(size_t)j*m_];
}

template <class F>
inline F& NumMat<F>::operator()(Int i, Int j)  { 
#if ( _DEBUGlevel_ >= 1 )
  if( i < 0 || i >= m_ ||
      j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << m_ << ", " << n_ << ")" << std::endl
      << "This index     ~ (" << i  << ", " << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return data_[i+(size_t)j*m_];
}

template <class F>
inline F* NumMat<F>::VecData(Int j)  const 
{ 
#if ( _DEBUGlevel_ >= 1 )
  if( j < 0 || j >= n_ ) {
    std::ostringstream msg;
    msg 
      << "Index is out of bound."  << std::endl
      << "Index bound    ~ (" << n_ << ")" << std::endl
      << "This index     ~ (" << j  << ")" << std::endl;
    ErrorHandling( msg.str().c_str() ); 
  }
#endif
  return &(data_[(size_t)j*m_]); 
}
/*
template <class F>
void NumMat<F>::Rearrange(Int blockNum) {
    Int cols = n_;
    Int rows = m_;
    Int r = rows % blockNum;
    Int blockSize = rows / blockNum;
    Int subBlockTotal = blockNum * cols;
    NumVec<F> buf(blockSize + 1);
    Int subBlockNum2, tmp;
    IntNumVec index2;
    if (r == 0){
        subBlockNum2 = blockNum * cols;
        index2.Resize(subBlockNum2);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = i / blockNum + (i % blockNum) * cols;
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Int cur_ptr = (i / blockNum)  * rows + (i % blockNum) * blockSize;
                Int target_ptr = (index2(i) / blockNum)  * rows + (index2(i) % blockNum) * blockSize;
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                tmp = index2(index2(i));
                index2(index2(i)) = index2(i);
                index2(i) = tmp;
            }
        }
        return;
    }
    subBlockNum2 = subBlockTotal - (rows % blockNum) * cols;
    index2.Resize(subBlockNum2);
    for (Int i = 0; i < subBlockNum2; i++){
        index2(i) = i / (blockNum - r) + (i % (blockNum - r)) * cols;
    }
    for(Int i = 0; i < subBlockNum2; i++){
        while(index2(i) != i){
            Int cur_ptr = (i / (blockNum - r))  * rows + (i % (blockNum - r)) * blockSize + (blockSize + 1) * r;
            Int target_ptr = (index2(i) / (blockNum - r))  * rows + (index2(i) % (blockNum - r)) * blockSize + (blockSize + 1) * r;
            memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
            memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
            memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
            tmp = index2(index2(i));
            index2(index2(i)) = index2(i);
            index2(i) = tmp;
        }
    }
    if (rows % blockNum == 0) {
        return;
    }
    Int subBlockNum1 = (rows % blockNum) * cols;
    IntNumVec index1(subBlockNum1);
    for (Int i = 0; i < subBlockNum1; i++){
        index1(i) = i / r + (i % r) * cols;
    }

    for(Int i = 0; i < subBlockNum1; i++){
        while(index1(i) != i){
            Int cur_ptr = (i / r)  * rows + (i % r) * (blockSize + 1);
            Int target_ptr = (index1(i) / r)  * rows + (index1(i) % r) * (blockSize + 1);
            memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * (blockSize + 1));
            memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * (blockSize + 1));
            memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) *(blockSize + 1));
            tmp = index1(index1(i));
            index1(index1(i)) = index1(i);
            index1(i) = tmp;
        }
    }
    Int largerBlockNum = 2;
    Int largerSubBlockNum = largerBlockNum * cols;
    IntNumVec largerBlockSize(largerBlockNum);
    largerBlockSize(0) = (rows%blockNum)*(blockSize+1);
    largerBlockSize(1) = rows-(rows%blockNum)*(blockSize+1);
    Int bufSize = largerBlockSize(0) > largerBlockSize(1) ? largerBlockSize(0) : largerBlockSize(1);
    buf.Resize(bufSize);
    Int ptr = 0;
    for (Int index = 0; index < cols; index++) {
        Int tmpCol = index % cols;             
        Int tmpRow = index / cols;             
        Int currentSubBlockEles = largerBlockSize(0);
        Int currentSubBlockPtr = tmpCol * (largerBlockSize(0) + largerBlockSize(1));
        if(ptr == currentSubBlockPtr){
            ptr += currentSubBlockEles;
            continue;
        }
        memcpy((void*)buf.Data(), (void*)(data_ + currentSubBlockPtr), sizeof(F) * currentSubBlockEles);
        for (Int k = currentSubBlockPtr - 1; k >= ptr; k--) {
            *(this->data_ + k + currentSubBlockEles) = *(this->data_ + k);
        }
        memcpy((void*)(this->data_ + ptr), buf.Data(), sizeof(F) * currentSubBlockEles);
        ptr += currentSubBlockEles;
    }
}





template <class F>
void NumMat<F>::RevertRearrange(Int blockNum) {
    Int cols = n_;
    Int rows = m_;
    Int r = rows % blockNum;
    Int blockSize = rows / blockNum;
    Int subBlockTotal = blockNum * cols;
    NumVec<F> buf;
    Int tmp;
    if (r != 0){
        Int largerBlockNum = 2;
        Int largerSubBlockNum = largerBlockNum * cols;
        IntNumVec largerBlockSize(largerBlockNum);
        largerBlockSize(0) = (rows%blockNum)*(blockSize+1);
        largerBlockSize(1) = rows-(rows%blockNum)*(blockSize+1);
        Int bufSize = largerBlockSize(0) > largerBlockSize(1) ? largerBlockSize(0) : largerBlockSize(1);
        buf.Resize(bufSize);
        Int ptr = cols * (largerBlockSize(0) + largerBlockSize(1)) - largerBlockSize(1);
        for(Int index = cols-1; index > 0; index--){
            Int tmpCol = index % cols;
            Int currentSubBlockEles = largerBlockSize(0);
            Int currentSubBlockPtr = tmpCol * largerBlockSize(0);
            memcpy((void*)buf.Data(), (void*)(data_ + currentSubBlockPtr), sizeof(F)*currentSubBlockEles);
            for(Int k = currentSubBlockPtr; k < ptr - currentSubBlockEles; k++){
                *(this->data_ + k) = *(this->data_ + k + currentSubBlockEles);
            }
            memcpy((void*)(this->data_ + ptr - currentSubBlockEles), buf.Data(), sizeof(F) * currentSubBlockEles);
            ptr -= largerBlockSize(0) + largerBlockSize(1);
        }
        buf.Resize(blockSize + 1);
        Int subBlockNum1 = (rows % blockNum) * cols;
        IntNumVec index1(subBlockNum1);
        for (Int i = 0; i < subBlockNum1; i++){
            index1(i) = (i % cols) * r + i / cols; 
        }
        for(Int i = 0; i < subBlockNum1; i++){
            while(index1(i) != i){
                Int cur_ptr = (i / r)  * rows + (i % r) * (blockSize + 1);
                Int target_ptr = (index1(i) / r)  * rows + (index1(i) % r) * (blockSize + 1);
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * (blockSize + 1));
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * (blockSize + 1));
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) *(blockSize + 1));
                tmp = index1(index1(i));
                index1(index1(i)) = index1(i);
                index1(i) = tmp;
            }
        }
        Int subBlockNum2 = subBlockTotal - (rows%blockNum)*cols ? subBlockTotal - (rows%blockNum)*cols :  rows*blockNum;
        IntNumVec index2(subBlockNum2);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = (i % cols) * (blockNum - r) + i / cols;
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Int cur_ptr = (i / (blockNum - r))  * rows + (i % (blockNum - r)) * blockSize + (blockSize + 1) * r;
                Int target_ptr = (index2(i) / (blockNum - r))  * rows + (index2(i) % (blockNum - r)) * blockSize + (blockSize + 1) * r;
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                tmp = index2(index2(i));
                index2(index2(i)) = index2(i);
                index2(i) = tmp;
            }
        }
    } else{
        Int subBlockNum2 = cols*blockNum;
        IntNumVec index2(subBlockNum2);
        buf.Resize(blockSize);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = (i % cols) * blockNum + i / cols;
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Int cur_ptr = (i / blockNum)  * rows + (i % blockNum) * blockSize + (blockSize + 1) * r;
                Int target_ptr = (index2(i) / blockNum)  * rows + (index2(i) % blockNum) * blockSize + (blockSize + 1) * r;
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                tmp = index2(index2(i));
                index2(index2(i)) = index2(i);
                index2(i) = tmp;
            }
        }
    }
}
*/

template <class F>
void NumMat<F>::Rearrange(Int blockNum) {
    typedef std::ptrdiff_t Idx;

    Int cols = n_;
    Int rows = m_;
    Int r = rows % blockNum;
    Int blockSize = rows / blockNum;
    Int subBlockTotal = blockNum * cols;
    NumVec<F> buf(blockSize + 1);
    Int subBlockNum2, tmp;
    IntNumVec index2;

    if (r == 0){
        subBlockNum2 = blockNum * cols;
        index2.Resize(subBlockNum2);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = i / blockNum + (i % blockNum) * cols;
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Idx cur_ptr    = (Idx)(i / blockNum) * rows
                               + (Idx)(i % blockNum) * blockSize;
                Idx target_ptr = (Idx)(index2(i) / blockNum) * rows
                               + (Idx)(index2(i) % blockNum) * blockSize;
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                tmp = index2(index2(i));
                index2(index2(i)) = index2(i);
                index2(i) = tmp;
            }
        }
        return;
    }

    subBlockNum2 = subBlockTotal - (rows % blockNum) * cols;
    index2.Resize(subBlockNum2);
    for (Int i = 0; i < subBlockNum2; i++){
        index2(i) = i / (blockNum - r) + (i % (blockNum - r)) * cols;
    }
    for(Int i = 0; i < subBlockNum2; i++){
        while(index2(i) != i){
            Idx cur_ptr    = (Idx)(i / (blockNum - r)) * rows
                           + (Idx)(i % (blockNum - r)) * blockSize
                           + (Idx)(blockSize + 1) * r;
            Idx target_ptr = (Idx)(index2(i) / (blockNum - r)) * rows
                           + (Idx)(index2(i) % (blockNum - r)) * blockSize
                           + (Idx)(blockSize + 1) * r;
            memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
            memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
            memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
            tmp = index2(index2(i));
            index2(index2(i)) = index2(i);
            index2(i) = tmp;
        }
    }
    if (rows % blockNum == 0) {
        return;
    }

    Int subBlockNum1 = (rows % blockNum) * cols;
    IntNumVec index1(subBlockNum1);
    for (Int i = 0; i < subBlockNum1; i++){
        index1(i) = i / r + (i % r) * cols;
    }
    for(Int i = 0; i < subBlockNum1; i++){
        while(index1(i) != i){
            Idx cur_ptr    = (Idx)(i / r) * rows
                           + (Idx)(i % r) * (blockSize + 1);
            Idx target_ptr = (Idx)(index1(i) / r) * rows
                           + (Idx)(index1(i) % r) * (blockSize + 1);
            memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * (blockSize + 1));
            memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * (blockSize + 1));
            memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * (blockSize + 1));
            tmp = index1(index1(i));
            index1(index1(i)) = index1(i);
            index1(i) = tmp;
        }
    }

    Int largerBlockNum = 2;
    IntNumVec largerBlockSize(largerBlockNum);
    largerBlockSize(0) = (rows % blockNum) * (blockSize + 1);
    largerBlockSize(1) = rows - (rows % blockNum) * (blockSize + 1);
    Int bufSize = largerBlockSize(0) > largerBlockSize(1) ? largerBlockSize(0) : largerBlockSize(1);
    buf.Resize(bufSize);

    Idx ptr = 0;
    for (Int index = 0; index < cols; index++) {
        Int tmpCol = index % cols;
        Int currentSubBlockEles = largerBlockSize(0);
        Idx currentSubBlockPtr  = (Idx)tmpCol * (largerBlockSize(0) + largerBlockSize(1));
        if(ptr == currentSubBlockPtr){
            ptr += currentSubBlockEles;
            continue;
        }
        memcpy((void*)buf.Data(), (void*)(data_ + currentSubBlockPtr), sizeof(F) * currentSubBlockEles);
        for (Idx k = currentSubBlockPtr - 1; k >= ptr; k--) {
            *(this->data_ + k + currentSubBlockEles) = *(this->data_ + k);
        }
        memcpy((void*)(this->data_ + ptr), buf.Data(), sizeof(F) * currentSubBlockEles);
        ptr += currentSubBlockEles;
    }
}



template <class F>
void NumMat<F>::RevertRearrange(Int blockNum) {
    typedef std::ptrdiff_t Idx;

    Int cols = n_;
    Int rows = m_;
    Int r = rows % blockNum;
    Int blockSize = rows / blockNum;
    Int subBlockTotal = blockNum * cols;
    NumVec<F> buf;
    Int tmp;

    if (r != 0){
        Int largerBlockNum = 2;
        IntNumVec largerBlockSize(largerBlockNum);
        largerBlockSize(0) = (rows % blockNum) * (blockSize + 1);
        largerBlockSize(1) = rows - (rows % blockNum) * (blockSize + 1);
        Int bufSize = largerBlockSize(0) > largerBlockSize(1) ? largerBlockSize(0) : largerBlockSize(1);
        buf.Resize(bufSize);

        Idx ptr = (Idx)cols * (largerBlockSize(0) + largerBlockSize(1))
                - largerBlockSize(1);
        for(Int index = cols - 1; index > 0; index--){
            Int tmpCol = index % cols;
            Int currentSubBlockEles = largerBlockSize(0);
            Idx currentSubBlockPtr  = (Idx)tmpCol * largerBlockSize(0);
            memcpy((void*)buf.Data(), (void*)(data_ + currentSubBlockPtr), sizeof(F) * currentSubBlockEles);
            for(Idx k = currentSubBlockPtr; k < ptr - currentSubBlockEles; k++){
                *(this->data_ + k) = *(this->data_ + k + currentSubBlockEles);
            }
            memcpy((void*)(this->data_ + ptr - currentSubBlockEles), buf.Data(), sizeof(F) * currentSubBlockEles);
            ptr -= (Idx)largerBlockSize(0) + largerBlockSize(1);
        }

        buf.Resize(blockSize + 1);
        Int subBlockNum1 = (rows % blockNum) * cols;
        IntNumVec index1(subBlockNum1);
        for (Int i = 0; i < subBlockNum1; i++){
            index1(i) = (i % cols) * r + i / cols;
        }
        for(Int i = 0; i < subBlockNum1; i++){
            while(index1(i) != i){
                Idx cur_ptr    = (Idx)(i / r) * rows
                               + (Idx)(i % r) * (blockSize + 1);
                Idx target_ptr = (Idx)(index1(i) / r) * rows
                               + (Idx)(index1(i) % r) * (blockSize + 1);
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * (blockSize + 1));
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * (blockSize + 1));
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * (blockSize + 1));
                tmp = index1(index1(i));
                index1(index1(i)) = index1(i);
                index1(i) = tmp;
            }
        }

        Int subBlockNum2 = subBlockTotal - (rows % blockNum) * cols
                         ? subBlockTotal - (rows % blockNum) * cols
                         : rows * blockNum;
        IntNumVec index2(subBlockNum2);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = (i % cols) * (blockNum - r) + i / cols;
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Idx cur_ptr    = (Idx)(i / (blockNum - r)) * rows
                               + (Idx)(i % (blockNum - r)) * blockSize
                               + (Idx)(blockSize + 1) * r;
                Idx target_ptr = (Idx)(index2(i) / (blockNum - r)) * rows
                               + (Idx)(index2(i) % (blockNum - r)) * blockSize
                               + (Idx)(blockSize + 1) * r;
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                tmp = index2(index2(i));
                index2(index2(i)) = index2(i);
                index2(i) = tmp;
            }
        }
    } else {
        Int subBlockNum2 = cols * blockNum;
        IntNumVec index2(subBlockNum2);
        buf.Resize(blockSize);
        for (Int i = 0; i < subBlockNum2; i++){
            index2(i) = (i % cols) * blockNum + i / cols;
        }
        for(Int i = 0; i < subBlockNum2; i++){
            while(index2(i) != i){
                Idx cur_ptr    = (Idx)(i / blockNum) * rows
                               + (Idx)(i % blockNum) * blockSize
                               + (Idx)(blockSize + 1) * r;
                Idx target_ptr = (Idx)(index2(i) / blockNum) * rows
                               + (Idx)(index2(i) % blockNum) * blockSize
                               + (Idx)(blockSize + 1) * r;
                memcpy((void *) buf.Data(), (void *)(data_ + cur_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + cur_ptr), (void *)(data_ + target_ptr), sizeof(F) * blockSize);
                memcpy((void *)(data_ + target_ptr), (void *)buf.Data(), sizeof(F) * blockSize);
                tmp = index2(index2(i));
                index2(index2(i)) = index2(i);
                index2(i) = tmp;
            }
        }
    }
}





template <class F>
void NumMat<F>::ExchangeCols(IntNumVec& colMap) {
    if (colMap.Size() != n_) {
        std::ostringstream msg;
        msg << "ERROR! Input mapping with wrong size, input size: " << colMap.Size()
            << "matrix column number: ( " << n_ << " )";
        ErrorHandling(msg.str().c_str());
    }
    int col_index = 0;
    while (col_index < colMap.Size()) {
        if (col_index == colMap[col_index]) {
            col_index++;
            continue;
        }
        NumVec<F> tmp_vec(m_);
        int tmp;
        while (col_index != colMap[col_index]) {
            if (col_index >= colMap.Size()){
                std::ostringstream msg;
                msg << "ERROR! Index of colMap exceeds the column dimension of the matrix." << std::endl 
                    << " Index: " <<  col_index << std::endl
                    << " Value: " << colMap[col_index] << std::endl
                    << " Matrix column number: ( " << n_ << " )" << std::endl << std::endl;
                ErrorHandling(msg.str().c_str());
            }
            memcpy(tmp_vec.Data(), this->VecData(colMap[col_index]), sizeof(F) * m_);
            tmp = colMap[colMap[col_index]];
            memcpy(this->VecData(colMap[col_index]), this->VecData(col_index), sizeof(F) * m_);
            colMap[colMap[col_index]] = colMap[col_index];
            memcpy(this->VecData(col_index), tmp_vec.Data(), sizeof(F) * m_);
            colMap[col_index] = tmp;
        }
    }
}








// *********************************************************************
// Utilities
// *********************************************************************

template <class F> inline void SetValue(NumMat<F>& M, F val)
{
  F *ptr = M.data_;
  for (size_t i=0; i < (size_t)M.m()*M.n(); i++) *(ptr++) = val;
}

template <class F> inline Real Energy(const NumMat<F>& M)
{
  Real sum = 0;
  F *ptr = M.data_;
  for (size_t i=0; i < (size_t) M.m()*M.n(); i++) 
    sum += std::abs(ptr[i]) * std::abs(ptr[i]);
  return sum;
}


template <class F> inline void
Transpose ( const NumMat<F>& A, NumMat<F>& B )
{
  if( A.m() != B.n() || A.n() != B.m() ){
    B.Resize( A.n(), A.m() );
  }

  F* Adata = A.Data();
  F* Bdata = B.Data();
  Int m = A.m(), n = A.n();

  for( Int i = 0; i < m; i++ ){
    for( Int j = 0; j < n; j++ ){
      Bdata[ j + (size_t)n*i ] = Adata[ i + (size_t)j*m ];
    }
  }


  return ;
}       

template <class F> inline void
Symmetrize( NumMat<F>& A )
{
  if( A.m() != A.n() ){
    ErrorHandling( "The matrix to be symmetrized should be a square matrix." );
  }

  NumMat<F> B;
  Transpose( A, B );

  F* Adata = A.Data();
  F* Bdata = B.Data();

  F  half = (F) 0.5;

  for( size_t i = 0; i <(size_t) A.m() * A.n(); i++ ){
    *Adata = half * (*Adata + *Bdata);
    Adata++; Bdata++;
  }
  return;
}
template <class F>
void printCpxM(NumMat<F> &Mat, std::ostream &out) {
    Int r1 = Mat.m();
    Int r2 = Mat.n();
    out << r1 << std::endl;
    out << r2 << std::endl;
    out <<"CPU"<<std::endl;
    for (int j = 0; j < r2; j++) {
       // out << "j=" << j << std::endl;
        for (Int i = 0; i < r1; i++) {
            out << std::fixed << std::setprecision(20) << Mat(i, j) << std::endl;
        }
    }
    out.flush();
}




} // namespace isdf

