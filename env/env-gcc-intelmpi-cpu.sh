#!/usr/bin/env bash

# ============================================================================
# LibISDF CPU environment
#
# Compiler:       GNU GCC/G++
# MPI:            Intel MPI
# Math library:   Intel oneMKL
# FFTW:           Double-precision FFTW with Intel MPI support
#
# Usage:
#
#   source env/env-gcc-intelmpi-cpu.sh
#
# This script must be sourced rather than executed.
# ============================================================================

# Require the script to be sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo
    echo "  source env/env-gcc-intelmpi-cpu.sh"
    echo
    exit 1
fi

# ============================================================================
# Installation paths
#
# Modify these paths when using another cluster or software installation.
# ============================================================================

export ONEAPI_ROOT=/public/software/intel/oneapi/2024.1.0

export MPI_DIR="${ONEAPI_ROOT}/mpi/2021.12"
export MKL_ROOT="${ONEAPI_ROOT}/mkl/2024.1"

# MKLROOT is the standard oneMKL environment-variable name.
# MKL_ROOT is used by the LibISDF Makefile.
export MKLROOT="${MKL_ROOT}"

# FFTW must be built with Intel MPI support.
export FFTW_DIR=/public/home/wuwt/lib/fftw

# ============================================================================
# Validate installation directories
# ============================================================================

_isdf_env_error=0

if [[ ! -d "${MPI_DIR}" ]]; then
    echo "ERROR: Intel MPI directory does not exist:"
    echo "  ${MPI_DIR}"
    _isdf_env_error=1
fi

if [[ ! -d "${MKL_ROOT}" ]]; then
    echo "ERROR: oneMKL directory does not exist:"
    echo "  ${MKL_ROOT}"
    _isdf_env_error=1
fi

if [[ ! -d "${FFTW_DIR}" ]]; then
    echo "ERROR: FFTW directory does not exist:"
    echo "  ${FFTW_DIR}"
    _isdf_env_error=1
fi

if [[ "${_isdf_env_error}" -ne 0 ]]; then
    unset _isdf_env_error
    return 1
fi

unset _isdf_env_error

# ============================================================================
# Load Intel MPI and oneMKL component environments
# ============================================================================

# Intel MPI environment
if [[ -f "${MPI_DIR}/env/vars.sh" ]]; then
    source "${MPI_DIR}/env/vars.sh"
else
    echo "WARNING: Intel MPI environment script was not found:"
    echo "  ${MPI_DIR}/env/vars.sh"
    echo "Using MPI_DIR directly."
fi

# oneMKL environment
if [[ -f "${MKL_ROOT}/env/vars.sh" ]]; then
    source "${MKL_ROOT}/env/vars.sh"
else
    echo "WARNING: oneMKL environment script was not found:"
    echo "  ${MKL_ROOT}/env/vars.sh"
    echo "Using MKL_ROOT directly."
fi

# Restore the values used by the LibISDF Makefile in case the component
# environment scripts changed them.
export MPI_DIR="${ONEAPI_ROOT}/mpi/2021.12"
export MKL_ROOT="${ONEAPI_ROOT}/mkl/2024.1"
export MKLROOT="${MKL_ROOT}"
export FFTW_DIR=/public/home/wuwt/lib/fftw

# ============================================================================
# FFTW library-directory detection
# ============================================================================

if [[ -d "${FFTW_DIR}/lib" ]]; then
    export FFTW_LIB_DIR="${FFTW_DIR}/lib"
elif [[ -d "${FFTW_DIR}/lib64" ]]; then
    export FFTW_LIB_DIR="${FFTW_DIR}/lib64"
else
    echo "ERROR: Neither of the following FFTW library directories exists:"
    echo "  ${FFTW_DIR}/lib"
    echo "  ${FFTW_DIR}/lib64"
    return 1
fi

# ============================================================================
# Search paths
# ============================================================================

# Put Intel MPI wrappers before any Open MPI or HPC-X wrappers.
export PATH="${MPI_DIR}/bin:${PATH}"

# Runtime library paths
export LD_LIBRARY_PATH="${MPI_DIR}/lib:${MKL_ROOT}/lib/intel64:${FFTW_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ============================================================================
# GNU compiler selection
# ============================================================================

# Select GCC/G++ as the underlying compilers used by the Intel MPI wrappers.
export I_MPI_CC=gcc
export I_MPI_CXX=g++

# The Makefile uses the generic mpicc/mpicxx wrappers.
export CC=mpicc
export CXX=mpicxx

# ============================================================================
# LibISDF Makefile configuration
# ============================================================================

# Use mpicc/mpicxx instead of mpiicc/mpiicpc or mpiicx/mpiicpx.
export MPI_FLAVOR=generic

# Select the Intel MPI BLACS library.
export MPI_IMPL=intelmpi

# Use the GNU OpenMP threading layer of oneMKL.
export MKL_THREADING=gnu_thread

# No additional Fortran runtime is required by default.
export FORTRAN_RT=none

# Select the CPU implementation.
export USE_GPU=0
export USE_GPUDIRECT=0

# Default build mode.
export COMPILE_MODE=release

# ============================================================================
# Default runtime thread configuration
#
# These values are conservative defaults. Override them in the Slurm script
# when more than one CPU core is allocated to each MPI process.
# ============================================================================

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

# ============================================================================
# Check required commands
# ============================================================================

_isdf_env_error=0

for _isdf_command in gcc g++ mpicc mpicxx mpirun; do
    if ! command -v "${_isdf_command}" >/dev/null 2>&1; then
        echo "ERROR: Required command was not found: ${_isdf_command}"
        _isdf_env_error=1
    fi
done

if [[ "${_isdf_env_error}" -ne 0 ]]; then
    unset _isdf_command
    unset _isdf_env_error
    return 1
fi

unset _isdf_command
unset _isdf_env_error

# ============================================================================
# Check FFTW MPI support
# ============================================================================

_isdf_fftw_header="missing"
_isdf_fftw_library="missing"

if [[ -f "${FFTW_DIR}/include/fftw3-mpi.h" ]]; then
    _isdf_fftw_header="found"
fi

if compgen -G "${FFTW_LIB_DIR}/libfftw3_mpi.so*" >/dev/null 2>&1 ||
   [[ -f "${FFTW_LIB_DIR}/libfftw3_mpi.a" ]]; then
    _isdf_fftw_library="found"
fi

# ============================================================================
# Print environment summary
# ============================================================================

echo "============================================================"
echo "LibISDF GNU/Intel-MPI CPU Environment"
echo "============================================================"
echo "ONEAPI_ROOT    = ${ONEAPI_ROOT}"
echo "MPI_DIR        = ${MPI_DIR}"
echo "MKL_ROOT       = ${MKL_ROOT}"
echo "FFTW_DIR       = ${FFTW_DIR}"
echo "FFTW_LIB_DIR   = ${FFTW_LIB_DIR}"
echo
echo "MPI_FLAVOR     = ${MPI_FLAVOR}"
echo "MPI_IMPL       = ${MPI_IMPL}"
echo "MKL_THREADING  = ${MKL_THREADING}"
echo "FORTRAN_RT     = ${FORTRAN_RT}"
echo
echo "I_MPI_CC       = ${I_MPI_CC}"
echo "I_MPI_CXX      = ${I_MPI_CXX}"
echo "CC             = ${CC}"
echo "CXX            = ${CXX}"
echo
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS = ${MKL_NUM_THREADS}"
echo "OMP_PROC_BIND   = ${OMP_PROC_BIND}"
echo "OMP_PLACES      = ${OMP_PLACES}"
echo "MKL_DYNAMIC     = ${MKL_DYNAMIC}"
echo
echo "gcc             = $(command -v gcc)"
echo "g++             = $(command -v g++)"
echo "mpicc           = $(command -v mpicc)"
echo "mpicxx          = $(command -v mpicxx)"
echo "mpirun          = $(command -v mpirun)"
echo
echo "FFTW MPI header = ${_isdf_fftw_header}"
echo "FFTW MPI library= ${_isdf_fftw_library}"
echo
echo "GCC version:"
gcc --version | head -n 1
echo
echo "G++ version:"
g++ --version | head -n 1
echo
echo "MPI version:"
mpirun --version 2>&1 | head -n 2
echo
echo "mpicxx backend:"
mpicxx -show 2>/dev/null || mpicxx --showme 2>/dev/null || true
echo "============================================================"

if [[ "${_isdf_fftw_header}" != "found" ]]; then
    echo "WARNING: fftw3-mpi.h was not found under:"
    echo "  ${FFTW_DIR}/include"
fi

if [[ "${_isdf_fftw_library}" != "found" ]]; then
    echo "WARNING: The double-precision FFTW MPI library was not found under:"
    echo "  ${FFTW_LIB_DIR}"
fi

unset _isdf_fftw_header
unset _isdf_fftw_library

