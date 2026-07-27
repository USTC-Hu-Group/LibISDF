
#ifdef GPU  
#pragma once
#include  "../common/environment.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace isdf {
extern cusolverDnHandle_t cusolverH;
namespace cusolver{

typedef  int               Int;

void Init(void);

void Destroy(void);

void Potrf( char uplo, Int n, double * A, Int lda );



} // namespace cusolver
} // namespace isdf

#endif
