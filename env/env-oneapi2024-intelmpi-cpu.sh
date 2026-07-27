#!/usr/bin/env bash

# =====================================================================
# ISDF CPU build environment
#
# Toolchain:
#   - Intel oneAPI C/C++ Compiler 2024.1
#   - Intel MPI 2021.12
#   - Intel oneMKL 2024.1
#   - Double-precision FFTW with MPI support
#
# Usage:
#   source env/env-oneapi2024-intelmpi-cpu.sh
#   make clean
#   make USE_GPU=0 -j8
#
# The user normally only needs to modify:
#   1. ONEAPI_MODULE_ROOT
#   2. FFTW_DIR
# =====================================================================


# ---------------- User-configurable paths ----------------

# Directory containing the Intel oneAPI modulefiles
export ONEAPI_MODULE_ROOT=/public/software/intel/oneapi/2024.1.0/modulefiles

# Double-precision FFTW installation built with Intel MPI support
export FFTW_DIR=/public/home/wuwt/lib/fftw


# ---------------- Load compiler, MPI, and MKL ----------------

module purge
module use "$ONEAPI_MODULE_ROOT"

# Load prerequisites first
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load oclfpga/2024.1.0

# Load the Intel compiler, Intel MPI, and oneMKL
module load compiler/2024.1.0
module load mpi/2021.12
module load mkl/2024.1

# GNU toolchain used by icx/icpx on this cluster
module load compiler/gcc/7.3.1


# ---------------- Dependency paths ----------------

# These variables are provided by the Intel MPI and MKL modules
export MPI_DIR=$I_MPI_ROOT
export MKL_ROOT=$MKLROOT


# ---------------- MPI compiler configuration ----------------

# Use Intel MPI mpicc/mpicxx wrappers
export MPI_FLAVOR=generic
export MPI_IMPL=intelmpi

# Select Intel oneAPI compilers as the wrapper backends
export I_MPI_CC=icx
export I_MPI_CXX=icpx


# ---------------- Library configuration ----------------

# Use sequential oneMKL
export MKL_THREADING=sequential

# No explicit Fortran runtime is required by default
export FORTRAN_RT=none

# Runtime library paths
export LD_LIBRARY_PATH=$MKL_ROOT/lib/intel64:$FFTW_DIR/lib:$MPI_DIR/lib/release:$MPI_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}


# ---------------- Remove conflicting environments ----------------

# Remove Open MPI settings
unset OMPI_CC OMPI_CXX OMPI_FC
unset OMPI_MCA_pml OMPI_MCA_btl

# Remove MPICH settings
unset MPICH_CC MPICH_CXX MPICH_FC

# Remove generic compiler overrides
unset CC CXX FC F77 F90

# This is a CPU-only environment
unset CUDA_DIR CUDA_MATH_DIR CUDA_HOME CUDA_PATH
unset NVCC NVCC_CCBIN


echo "============================================================"
echo "ISDF CPU Environment"
echo "============================================================"
echo "MPI_DIR       = $MPI_DIR"
echo "MKL_ROOT      = $MKL_ROOT"
echo "FFTW_DIR      = $FFTW_DIR"
echo "MPI_IMPL      = $MPI_IMPL"
echo "I_MPI_CC      = $I_MPI_CC"
echo "I_MPI_CXX     = $I_MPI_CXX"
echo
echo "mpicc         = $(command -v mpicc)"
echo "mpicxx        = $(command -v mpicxx)"
echo "icx           = $(command -v icx)"
echo "icpx          = $(command -v icpx)"
echo
echo "mpicxx backend:"
mpicxx -show
echo "============================================================"

