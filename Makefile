# =====================================================================
# LibISDF Makefile
# =====================================================================
#
# Supported configurations:
#   - CPU and NVIDIA CUDA GPU builds
#   - Intel MPI, Open MPI, and MPICH
#   - Intel oneAPI, Intel Classic, and GNU compilers
#   - Intel oneMKL, ScaLAPACK, BLACS, and double-precision FFTW
#
# Repository layout:
#
#   include/                         Public and internal headers
#   src/common/                      Common implementation
#   src/cpu/                         CPU implementation
#   src/cuda/                        CUDA implementation
#   examples/si8/readpsi.cpp     Example program
#   examples/si8/data/           Example input data
#   env/                             Environment scripts
#   scripts/                         Slurm job scripts
#
# Quick start:
#
#   CPU build:
#
#     source env/env-oneapi2024-intelmpi-cpu.sh
#     make USE_GPU=0 info
#     make clean
#     make USE_GPU=0 -j8
#   CPU Slurm submission:
#   
#     sbatch scripts/run-cpu.slurm
#
#   GPU build:
#
#     source env/env-gcc73-hpcx-gpu.sh
#     make USE_GPU=1 USE_GPUDIRECT=0 CUDA_ARCH=80 info
#     make clean
#     make USE_GPU=1 USE_GPUDIRECT=0 CUDA_ARCH=80 -j8
#
#   GPU build with CUDA-aware MPI:
#
#     make clean
#     make USE_GPU=1 USE_GPUDIRECT=1 CUDA_ARCH=80 -j8
#
#
#   GPU Slurm submission:
#   
#     sbatch scripts/run-gpu.slurm
# Required environment variables:
#
#   MPI_DIR       MPI installation root
#   FFTW_DIR      FFTW installation root
#   MKL_ROOT      Intel oneMKL installation root
#
# Additional variables required for GPU builds:
#
#   CUDA_DIR      CUDA Toolkit root
#   NVCC          Path to nvcc; defaults to $(CUDA_DIR)/bin/nvcc
#
# Optional variables:
#
#   CUDA_MATH_DIR Additional CUDA math-library root, for example the
#                 math_libs directory provided by NVIDIA HPC SDK
#
# =====================================================================


# =====================================================================
# User-configurable options
# =====================================================================

# Build mode: release | debug
COMPILE_MODE ?= release

# Enable the CUDA implementation: 0 | 1
USE_GPU ?= 0

# Enable CUDA-aware MPI code paths: 0 | 1
USE_GPUDIRECT ?= 0

# MPI wrapper family: auto | intel | generic
MPI_FLAVOR ?= auto

# MPI implementation used for MKL BLACS:
# auto | intelmpi | openmpi | mpich
MPI_IMPL ?= auto

# Additional Fortran runtime:
# auto | ifcore | gfortran | none
FORTRAN_RT ?= none

# MKL threading layer:
# sequential | intel_thread | gnu_thread
MKL_THREADING ?= sequential

# CUDA compute capability:
# 70 = V100, 80 = A100, 90 = H100
CUDA_ARCH ?= 80

# C++ language standard
CXX_STANDARD ?= c++14

# Number of MPI processes used by "make run"
NPROCS ?= 2

# Arguments passed to the si8 example
RUN_ARGS ?= examples/si8/data

# =====================================================================
# Allow "make clean" and "make help" without loading an environment
# =====================================================================

SKIP_CONFIG_CHECK := 0

ifneq ($(strip $(MAKECMDGOALS)),)
  ifeq ($(strip $(filter-out clean help,$(MAKECMDGOALS))),)
    SKIP_CONFIG_CHECK := 1
  endif
endif


# =====================================================================
# Required environment-variable checks
# =====================================================================

ifeq ($(SKIP_CONFIG_CHECK),0)

  ifndef MPI_DIR
    $(error MPI_DIR is not set. Source an environment script or export MPI_DIR=/path/to/mpi)
  endif

  ifndef FFTW_DIR
    $(error FFTW_DIR is not set. Source an environment script or export FFTW_DIR=/path/to/fftw)
  endif

  ifndef MKL_ROOT
    $(error MKL_ROOT is not set. Source an environment script or export MKL_ROOT=/path/to/mkl)
  endif

  ifeq ($(USE_GPU),1)
    ifndef CUDA_DIR
      $(error CUDA_DIR is not set. It is required when USE_GPU=1)
    endif

    NVCC ?= $(CUDA_DIR)/bin/nvcc
  endif

endif


# =====================================================================
# Build mode
# =====================================================================

ifeq ($(COMPILE_MODE),release)

  COMPILE_DEF  := -DRELEASE
  COMPILE_FLAG := -O2 -g

else ifeq ($(COMPILE_MODE),debug)

  COMPILE_DEF  := -DDEBUG=1
  COMPILE_FLAG := -O0 -g

else

  $(error Invalid COMPILE_MODE='$(COMPILE_MODE)'. Use release or debug)

endif


# =====================================================================
# MPI wrapper selection
# =====================================================================
#
# MPI_FLAVOR=intel:
#   Prefer the oneAPI wrappers mpiicpx/mpiicx. Fall back to the Intel
#   Classic wrappers mpiicpc/mpiicc.
#
# MPI_FLAVOR=generic:
#   Use mpicxx/mpicc. The underlying compiler can be selected using
#   implementation-specific environment variables such as:
#
#     I_MPI_CXX=icpx
#     I_MPI_CC=icx
#
#     OMPI_CXX=g++
#     OMPI_CC=gcc
#
# MPI_FLAVOR controls the wrapper family. MPI_IMPL controls the MKL
# BLACS library and must identify the actual MPI implementation.
# =====================================================================
#
#ifeq ($(MPI_FLAVOR),auto)
#
#  ifneq ($(shell command -v mpiicpx 2>/dev/null),)
#    MPI_FLAVOR := intel
#  else ifneq ($(shell command -v mpiicpc 2>/dev/null),)
#    MPI_FLAVOR := intel
#  else
#    MPI_FLAVOR := generic
#  endif
#
#endif
#
#
#ifeq ($(MPI_FLAVOR),intel)
#
#  ifneq ($(shell command -v mpiicpx 2>/dev/null),)
#    MPI_CXX ?= mpiicpx
#    MPI_CC  ?= mpiicx
#  else
#    MPI_CXX ?= mpiicpc
#    MPI_CC  ?= mpiicc
#  endif
#
#else ifeq ($(MPI_FLAVOR),generic)
#
#  MPI_CXX ?= mpicxx
#  MPI_CC  ?= mpicc
#
#else
#
#  $(error Invalid MPI_FLAVOR='$(MPI_FLAVOR)'. Use auto, intel, or generic)
#
#endif

MPI_FLAVOR_RESOLVED := $(strip $(MPI_FLAVOR))

ifeq ($(MPI_FLAVOR_RESOLVED),auto)

  ifneq ($(shell command -v mpiicpx 2>/dev/null),)
    MPI_FLAVOR_RESOLVED := intel
  else ifneq ($(shell command -v mpiicpc 2>/dev/null),)
    MPI_FLAVOR_RESOLVED := intel
  else
    MPI_FLAVOR_RESOLVED := generic
  endif

endif


ifeq ($(MPI_FLAVOR_RESOLVED),intel)

  ifneq ($(shell command -v mpiicpx 2>/dev/null),)
    MPI_CXX ?= mpiicpx
    MPI_CC  ?= mpiicx
  else
    MPI_CXX ?= mpiicpc
    MPI_CC  ?= mpiicc
  endif

else ifeq ($(MPI_FLAVOR_RESOLVED),generic)

  MPI_CXX ?= mpicxx
  MPI_CC  ?= mpicc

else

  $(error Invalid MPI_FLAVOR='$(MPI_FLAVOR)'. Use auto, intel, or generic)

endif




# =====================================================================
# MPI implementation detection
# =====================================================================
#
# MPI_IMPL determines the MKL BLACS library:
#
#   intelmpi -> -lmkl_blacs_intelmpi_lp64
#   openmpi  -> -lmkl_blacs_openmpi_lp64
#   mpich    -> -lmkl_blacs_mpich_lp64
#
# Explicitly setting MPI_IMPL in the environment script is recommended.
# =====================================================================

ifeq ($(MPI_IMPL),auto)

  ifeq ($(MPI_FLAVOR),intel)

    MPI_IMPL := intelmpi

  else ifneq ($(strip $(I_MPI_ROOT)),)

    MPI_IMPL := intelmpi

  else

    MPI_SHOW    := $(shell $(MPI_CXX) -show 2>/dev/null)
    MPI_VERSION := $(shell $(MPI_CXX) --showme:version 2>/dev/null)

    ifneq ($(findstring Open MPI,$(MPI_VERSION)),)
      MPI_IMPL := openmpi
    else ifneq ($(findstring open-mpi,$(MPI_SHOW)),)
      MPI_IMPL := openmpi
    else ifneq ($(findstring openmpi,$(MPI_SHOW)),)
      MPI_IMPL := openmpi
    else ifneq ($(findstring /ompi/,$(MPI_SHOW)),)
      MPI_IMPL := openmpi
    else ifneq ($(findstring MPICH,$(MPI_VERSION)),)
      MPI_IMPL := mpich
    else ifneq ($(findstring mpich,$(MPI_SHOW)),)
      MPI_IMPL := mpich
    else ifeq ($(SKIP_CONFIG_CHECK),0)
      $(error Cannot detect the MPI implementation. Set MPI_IMPL=intelmpi, openmpi, or mpich explicitly)
    endif

  endif

endif


ifneq ($(filter $(MPI_IMPL),intelmpi openmpi mpich),$(MPI_IMPL))
  ifeq ($(SKIP_CONFIG_CHECK),0)
    $(error Invalid MPI_IMPL='$(MPI_IMPL)'. Use intelmpi, openmpi, or mpich)
  endif
endif


# =====================================================================
# Fortran runtime
# =====================================================================
#
# Some externally built FFTW, ScaLAPACK, or MPI libraries may require a
# Fortran runtime. Use FORTRAN_RT=none if no additional runtime library
# is required.
# =====================================================================

ifeq ($(FORTRAN_RT),auto)

  ifeq ($(MPI_FLAVOR),intel)

    ifeq ($(findstring g++,$(I_MPI_CXX))$(findstring gcc,$(I_MPI_CC)),)
      FORTRAN_RT := ifcore
    else
      FORTRAN_RT := gfortran
    endif

  else

    FORTRAN_RT := gfortran

  endif

endif


ifeq ($(FORTRAN_RT),ifcore)

  FORTRAN_LIB := -lifcore

else ifeq ($(FORTRAN_RT),gfortran)

  FORTRAN_LIB := -lgfortran

else ifeq ($(FORTRAN_RT),none)

  FORTRAN_LIB :=

else

  $(error Invalid FORTRAN_RT='$(FORTRAN_RT)'. Use auto, ifcore, gfortran, or none)

endif


# =====================================================================
# Intel oneMKL library directory
# =====================================================================

ifeq ($(SKIP_CONFIG_CHECK),0)

  ifneq ($(wildcard $(MKL_ROOT)/lib/intel64/libmkl_core.*),)

    MKL_LIB_DIR := $(MKL_ROOT)/lib/intel64

  else ifneq ($(wildcard $(MKL_ROOT)/lib/libmkl_core.*),)

    MKL_LIB_DIR := $(MKL_ROOT)/lib

  else

    $(error Cannot find MKL libraries under $(MKL_ROOT)/lib/intel64 or $(MKL_ROOT)/lib)

  endif

endif


# =====================================================================
# MKL threading layer
# =====================================================================

ifeq ($(MKL_THREADING),sequential)

  MKL_THREAD_LIB := -lmkl_sequential

else ifeq ($(MKL_THREADING),intel_thread)

  MKL_THREAD_LIB := -lmkl_intel_thread -liomp5

else ifeq ($(MKL_THREADING),gnu_thread)

  MKL_THREAD_LIB := -lmkl_gnu_thread -lgomp

else

  $(error Invalid MKL_THREADING='$(MKL_THREADING)')

endif


# =====================================================================
# MKL ScaLAPACK and BLACS libraries
# =====================================================================

MKL_BLACS_LIB := -lmkl_blacs_$(MPI_IMPL)_lp64

MKL_LIB := -L$(MKL_LIB_DIR) \
           -lmkl_scalapack_lp64 \
           -lmkl_intel_lp64 \
           $(MKL_THREAD_LIB) \
           -lmkl_core \
           $(MKL_BLACS_LIB) \
           -lpthread \
           -lm \
           -ldl


# =====================================================================
# FFTW library directory
# =====================================================================

ifeq ($(SKIP_CONFIG_CHECK),0)

  ifneq ($(wildcard $(FFTW_DIR)/lib/libfftw3.*),)

    FFTW_LIB_DIR := $(FFTW_DIR)/lib

  else ifneq ($(wildcard $(FFTW_DIR)/lib64/libfftw3.*),)

    FFTW_LIB_DIR := $(FFTW_DIR)/lib64

  else

    $(error Cannot find libfftw3 under $(FFTW_DIR)/lib or $(FFTW_DIR)/lib64)

  endif

endif


# =====================================================================
# Include directories
# =====================================================================

ISDF_INCLUDE := -Iinclude \
                -Iinclude/common

MPI_INCLUDE  := -I$(MPI_DIR)/include

FFTW_INCLUDE := -I$(FFTW_DIR)/include \
                -I$(MKL_ROOT)/include

INCLUDES := $(ISDF_INCLUDE) \
            $(MPI_INCLUDE) \
            $(FFTW_INCLUDE)

ifeq ($(USE_GPU),1)

  INCLUDES += -Iinclude/cpu \
              -Iinclude/cuda

else

  INCLUDES += -Iinclude/cpu

endif


# =====================================================================
# External libraries
# =====================================================================

FFTW_LIB := -L$(FFTW_LIB_DIR) \
            -lfftw3_mpi \
            -lfftw3

LIBS := $(MKL_LIB) \
        $(FORTRAN_LIB) \
        $(FFTW_LIB)


# =====================================================================
# CPU/GPU compiler configuration
# =====================================================================
#
# Design rule:
#
#   - C and C++ files are compiled with MPI compiler wrappers.
#   - CUDA .cu files are compiled with nvcc.
#
# The nvcc host compiler must be a real compiler, not an MPI wrapper.
# Override NVCC_CCBIN from the environment when necessary.
# =====================================================================

ifeq ($(USE_GPU),1)

  CUDA_DEF := -DGPU

  ifeq ($(USE_GPUDIRECT),1)
    CUDA_DEF += -DGPUDIRECT
  endif

  CUDA_INC := -I$(CUDA_DIR)/include

  NVCC_CCBIN ?= g++

  # Convert each host flag into a separate nvcc -Xcompiler argument.
  NVCC_HOST_FLAGS := $(foreach flag,$(COMPILE_FLAG),-Xcompiler $(flag))

  NVCCFLAG := -ccbin=$(NVCC_CCBIN) \
              -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
              $(CUDA_DEF) \
              $(INCLUDES) \
              $(CUDA_INC) \
              $(NVCC_HOST_FLAGS) \
              -std=$(CXX_STANDARD)

  CXX := $(MPI_CXX)
  CC  := $(MPI_CC)

  CXXFLAGS := $(COMPILE_FLAG) \
              $(INCLUDES) \
              $(CUDA_DEF) \
              $(CUDA_INC) \
              -std=$(CXX_STANDARD)

  CFLAGS := $(COMPILE_FLAG) \
            $(INCLUDES) \
            $(CUDA_DEF) \
            $(CUDA_INC)

  # CUDA Toolkit libraries.
  CUDA_LIB_DIRS := -L$(CUDA_DIR)/lib64 \
                   -L$(CUDA_DIR)/lib64/stubs

  # CUDA_MATH_DIR is optional. It is commonly provided by NVIDIA HPC SDK.
  ifneq ($(strip $(CUDA_MATH_DIR)),)
    CUDA_LIB_DIRS += -L$(CUDA_MATH_DIR)/lib
  endif

  CUDA_LIB := $(CUDA_LIB_DIRS) \
              -lcufft \
              -lcublas \
              -lcusolver \
              -lcudart \
              -lcuda

  LIBS += $(CUDA_LIB)

  LOADER := $(MPI_CXX)

else

  CXX := $(MPI_CXX)
  CC  := $(MPI_CC)

  CXXFLAGS := $(COMPILE_FLAG) \
              $(INCLUDES) \
              -std=$(CXX_STANDARD)

  CFLAGS := $(COMPILE_FLAG) \
            $(INCLUDES)

  LOADER := $(MPI_CXX)

endif


CPPDEFS := $(COMPILE_DEF)
CCDEFS  := $(COMPILE_DEF)


# =====================================================================
# Build tools
# =====================================================================

AR      ?= ar
RANLIB  ?= ranlib
RM      ?= rm
MPIRUN  ?= mpirun

ARFLAGS := rcs
RMFLAGS ?= -f


# =====================================================================
# Library source files and object files
# =====================================================================

COMMON_SRCS_CPP := $(wildcard src/common/*.cpp)
COMMON_OBJS_CPP := $(COMMON_SRCS_CPP:.cpp=.o)

CPU_SRCS_CPP := $(wildcard src/cpu/*.cpp)
CPU_OBJS_CPP := $(CPU_SRCS_CPP:.cpp=.o)


ifeq ($(USE_GPU),1)

  CUDA_SRCS_CPP := $(wildcard src/cuda/*.cpp)
  CUDA_SRCS_CU  := $(wildcard src/cuda/*.cu)

  CUDA_OBJS_CPP := $(CUDA_SRCS_CPP:.cpp=.o)
  CUDA_OBJS_CU  := $(CUDA_SRCS_CU:.cu=.o)

  LIB_OBJS := $(COMMON_OBJS_CPP) \
              $(CPU_OBJS_CPP) \
              $(CUDA_OBJS_CPP) \
              $(CUDA_OBJS_CU)

else

  LIB_OBJS := $(COMMON_OBJS_CPP) \
              $(CPU_OBJS_CPP)

endif


# =====================================================================
# Example program
# =====================================================================

EXAMPLE_DIR  := examples/si8
READPSI_SRC  := $(EXAMPLE_DIR)/readpsi.cpp
READPSI_OBJ  := $(EXAMPLE_DIR)/readpsi.o
EXAMPLE_DATA := $(EXAMPLE_DIR)/data


# =====================================================================
# Build targets
# =====================================================================

TARGET_LIB  := libisdf.a
TARGET_EXEC := readpsi


# =====================================================================
# Main targets
# =====================================================================

.PHONY: all lib example info clean run help

.DEFAULT_GOAL := all

all: info $(TARGET_LIB) $(TARGET_EXEC)

lib: info $(TARGET_LIB)

example: $(TARGET_EXEC)


# =====================================================================
# Configuration summary
# =====================================================================

info:
	@echo "================ LibISDF Build Configuration ================"
	@echo "  COMPILE_MODE   = $(COMPILE_MODE)"
	@echo "  USE_GPU        = $(USE_GPU)"
	@echo "  USE_GPUDIRECT  = $(USE_GPUDIRECT)"
	@echo "  MPI_FLAVOR     = $(MPI_FLAVOR)"
	@echo "  MPI_IMPL       = $(MPI_IMPL)"
	@echo "  MPI_CXX        = $(MPI_CXX)"
	@echo "  MPI_CC         = $(MPI_CC)"
	@echo "  FORTRAN_RT     = $(FORTRAN_RT)"
	@echo "  FORTRAN_LIB    = $(FORTRAN_LIB)"
	@echo "  MKL_LIB_DIR    = $(MKL_LIB_DIR)"
	@echo "  MKL_THREADING  = $(MKL_THREADING)"
	@echo "  MKL_BLACS_LIB  = $(MKL_BLACS_LIB)"
	@echo "  FFTW_LIB_DIR   = $(FFTW_LIB_DIR)"
	@echo "  CXX_STANDARD   = $(CXX_STANDARD)"
ifeq ($(USE_GPU),1)
	@echo "  CUDA_ARCH      = sm_$(CUDA_ARCH)"
	@echo "  CUDA_DIR       = $(CUDA_DIR)"
	@echo "  CUDA_MATH_DIR  = $(CUDA_MATH_DIR)"
	@echo "  NVCC           = $(NVCC)"
	@echo "  NVCC_CCBIN     = $(NVCC_CCBIN)"
endif
	@echo "  MPI_DIR        = $(MPI_DIR)"
	@echo "  MKL_ROOT       = $(MKL_ROOT)"
	@echo "  FFTW_DIR       = $(FFTW_DIR)"
	@echo "  TARGET_LIB     = $(TARGET_LIB)"
	@echo "  TARGET_EXEC    = $(TARGET_EXEC)"
	@echo "  READPSI_SRC    = $(READPSI_SRC)"
	@echo "  EXAMPLE_DATA   = $(EXAMPLE_DATA)"
	@echo "=============================================================="


# =====================================================================
# Library and example build rules
# =====================================================================

$(TARGET_LIB): $(LIB_OBJS)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@


$(TARGET_EXEC): $(READPSI_OBJ) $(TARGET_LIB)
	$(LOADER) -o $@ $(READPSI_OBJ) $(TARGET_LIB) $(LIBS)


# =====================================================================
# Generic compilation rules
# =====================================================================

# C++ sources
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CPPDEFS) $< -o $@


# C sources
%.o: %.c
	$(CC) -c $(CFLAGS) $(CCDEFS) $< -o $@


# CUDA sources
ifeq ($(USE_GPU),1)

%.o: %.cu
	$(NVCC) -c $(NVCCFLAG) $(CPPDEFS) $< -o $@

else

%.o: %.cu
	@echo "*** Error: CUDA source '$<' cannot be compiled with USE_GPU=0."
	@echo "    Rebuild with USE_GPU=1."
	@exit 1

endif


# =====================================================================
# Explicit dependencies for the si8 example
# =====================================================================

$(READPSI_OBJ): $(READPSI_SRC) \
                 include/isdf.hpp \
                 $(wildcard include/common/*.hpp) \
                 $(wildcard include/cpu/*.hpp) \
                 $(wildcard include/cuda/*.hpp)
	$(CXX) -c $(CXXFLAGS) $(CPPDEFS) $< -o $@


# =====================================================================
# Clean and run targets
# =====================================================================

clean:
	$(RM) $(RMFLAGS) \
	    src/common/*.o \
	    src/cpu/*.o \
	    src/cuda/*.o \
	    $(READPSI_OBJ) \
	    $(TARGET_EXEC) \
	    $(TARGET_LIB)


run: $(TARGET_EXEC)
	$(MPIRUN) -np $(NPROCS) ./$(TARGET_EXEC) $(RUN_ARGS)


# =====================================================================
# Help
# =====================================================================

help:
	@echo "LibISDF Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make"
	@echo "      Build libisdf.a and the si8 example."
	@echo ""
	@echo "  make lib"
	@echo "      Build libisdf.a only."
	@echo ""
	@echo "  make example"
	@echo "      Build the si8 example."
	@echo ""
	@echo "  make info"
	@echo "      Print the resolved build configuration."
	@echo ""
	@echo "  make clean"
	@echo "      Remove generated object files, libraries, and executables."
	@echo ""
	@echo "  make run"
	@echo "      Run the si8 example with MPI."
	@echo ""
	@echo "Common variables:"
	@echo "  COMPILE_MODE=release|debug"
	@echo "  USE_GPU=0|1"
	@echo "  USE_GPUDIRECT=0|1"
	@echo "  CUDA_ARCH=70|80|90"
	@echo "  MPI_FLAVOR=auto|intel|generic"
	@echo "  MPI_IMPL=intelmpi|openmpi|mpich"
	@echo "  FORTRAN_RT=auto|ifcore|gfortran|none"
	@echo "  MKL_THREADING=sequential|intel_thread|gnu_thread"
	@echo "  NVCC_CCBIN=g++|icpx"
	@echo "  NPROCS=N"
	@echo "  RUN_ARGS=/path/to/input/data"
	@echo ""
	@echo "Examples:"
	@echo "  make USE_GPU=0 -j8"
	@echo "  make USE_GPU=1 USE_GPUDIRECT=0 CUDA_ARCH=80 -j8"
	@echo "  make USE_GPU=1 USE_GPUDIRECT=1 CUDA_ARCH=80 -j8"
	@echo "  make NPROCS=2 run"
	@echo "  make NPROCS=2 RUN_ARGS=examples/si8/data run"
