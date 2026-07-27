

#ifdef GPU   
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cufft.h>

char *cublasGetErrorString(cublasStatus_t error);
char *cufftGetErrorString(cufftResult error);
#endif
