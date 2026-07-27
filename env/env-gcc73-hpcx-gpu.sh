#!/usr/bin/env bash

# =====================================================================
# ISDF GPU build environment
#
# Toolchain:
#   - GCC 7.3.1
#   - NVIDIA HPC-X 2.19 / Open MPI
#   - CUDA 12.4
#   - Intel oneMKL 2024.1
#   - Double-precision FFTW
#
# Usage:
#   source env/env-gcc73-hpcx-gpu.sh
#
# Build without CUDA-aware MPI:
#   make clean
#   make USE_GPU=1 USE_GPUDIRECT=0 CUDA_ARCH=80 -j8
#
# Build with CUDA-aware MPI:
#   make clean
#   make USE_GPU=1 USE_GPUDIRECT=1 CUDA_ARCH=80 -j8
#
# The user normally only needs to modify:
#   1. NVHPC_ROOT
#   2. FFTW_DIR
#   3. MKL_ENV_SCRIPT
# =====================================================================


export NVHPC_ROOT=/public/software/nvhpc/24.5/Linux_x86_64/24.5

export FFTW_DIR=/public/software/mathlib/fftw/nvhpc/24.5/3.3.10

export MKL_ENV_SCRIPT=/public/software/intel/oneapi/2024.1.0/mkl/2024.1/env/vars.sh


export HPCX=$NVHPC_ROOT/comm_libs/12.4/hpcx/hpcx-2.19
export CUDA_DIR=$NVHPC_ROOT/cuda/12.4
export CUDA_MATH_DIR=$NVHPC_ROOT/math_libs/12.4/targets/x86_64-linux

export NVCC=$CUDA_DIR/bin/nvcc



module purge

module load nvhpc/24.5
module load compiler/gcc/7.3.1



source "$HPCX/hpcx-init.sh"
hpcx_load

export MPI_DIR=$HPCX/ompi



source "$MKL_ENV_SCRIPT" intel64

export MKL_ROOT=$MKLROOT


export MPI_FLAVOR=generic
export MPI_IMPL=openmpi

export OMPI_CC=gcc
export OMPI_CXX=g++
export OMPI_FC=gfortran



export NVCC_CCBIN=g++



export MKL_THREADING=sequential

export FORTRAN_RT=gfortran

export LIBRARY_PATH=$CUDA_DIR/lib64:$CUDA_DIR/lib64/stubs:$CUDA_MATH_DIR/lib${LIBRARY_PATH:+:$LIBRARY_PATH}

export LD_LIBRARY_PATH=$MPI_DIR/lib:$MKL_ROOT/lib/intel64:$FFTW_DIR/lib:$CUDA_DIR/lib64:$CUDA_MATH_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}



unset I_MPI_CC I_MPI_CXX I_MPI_FC I_MPI_F90 I_MPI_ROOT

unset MPICH_CC MPICH_CXX MPICH_FC

unset CC CXX FC F77 F90


export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib

unset UCX_TLS
unset UCX_MEMTYPE_CACHE



echo "============================================================"
echo "ISDF GPU Environment"
echo "============================================================"
echo "NVHPC_ROOT    = $NVHPC_ROOT"
echo "HPCX          = $HPCX"
echo "MPI_DIR       = $MPI_DIR"
echo "MKL_ROOT      = $MKL_ROOT"
echo "FFTW_DIR      = $FFTW_DIR"
echo "CUDA_DIR      = $CUDA_DIR"
echo "CUDA_MATH_DIR = $CUDA_MATH_DIR"
echo "MPI_IMPL      = $MPI_IMPL"
echo "NVCC          = $NVCC"
echo "NVCC_CCBIN    = $NVCC_CCBIN"
echo
echo "mpicc         = $(command -v mpicc)"
echo "mpicxx        = $(command -v mpicxx)"
echo "gcc           = $(command -v gcc)"
echo "g++           = $(command -v g++)"
echo "nvcc          = $(command -v nvcc)"
echo
echo "mpicxx backend:"
mpicxx --showme:command 2>/dev/null || mpicxx -show
echo
echo "CUDA-aware Open MPI:"
ompi_info --parsable --all 2>/dev/null |
    grep mpi_built_with_cuda_support:value ||
    echo "Unable to determine CUDA-aware MPI support"
echo "============================================================"

