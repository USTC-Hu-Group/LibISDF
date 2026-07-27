LibISDF

LibISDF is a distributed-memory library for interpolative separable
density fitting (ISDF). It is designed primarily for integration with
plane-wave density functional theory programs.

The library provides CPU and NVIDIA CUDA GPU implementations. MPI is
used for distributed-memory parallelism, and the CPU/GPU implementation
is selected at compile time through the Makefile.

The current interface supports double-precision real-valued
wavefunctions at the Gamma point.

## Features

- MPI-based distributed-memory parallelism
- CPU implementation with MPI or ScaLAPACK backends
- Multi-GPU implementation with CUDA
- Optional CUDA-aware MPI communication
- QRCP and K-means interpolation-point selection
- Intel oneMKL, ScaLAPACK, and BLACS
- Double-precision FFTW with MPI support
- Intel MPI, Open MPI, and MPICH
- Intel oneAPI, Intel Classic, and GNU compilers

## Repository structure

```text
LibISDF/
├── README.md
├── Makefile
├── include/
│   ├── isdf.hpp
│   ├── common/
│   ├── cpu/
│   └── cuda/
├── src/
│   ├── common/
│   ├── cpu/
│   └── cuda/
├── env/
│   ├── env-oneapi2024-intelmpi-cpu.sh
│   ├── env-gcc73-hpcx-gpu.sh
│   └── env-gcc-intelmpi-cpu.sh
├── scripts/
│   ├── run-cpu.slurm
│   └── run-gpu.slurm
└── examples/
    └── si8/
        ├── README.md
        ├── readpsi.cpp
        └── data/
            ├── psirow0
            └── psirow1
```

The files `libisdf.a`, `readpsi`, and `*.o` are generated during the
build.

## Requirements

Both CPU and GPU builds require:

- A C++14-compatible compiler
- MPI
- Intel oneMKL with ScaLAPACK and BLACS
- Double-precision FFTW with MPI support

GPU builds additionally require:

- NVIDIA CUDA Toolkit
- cuBLAS, cuFFT, and cuSOLVER
- A CUDA-compatible host compiler
- CUDA-aware MPI when `USE_GPUDIRECT=1`

The FFTW MPI library must be built against the same MPI implementation
used to build LibISDF. For example, an Intel MPI build must use an
Intel MPI-compatible FFTW installation.

## Environment configuration

Example environment scripts are provided under `env/`.

For CPU builds, choose an environment script matching the compiler and
MPI implementation:

```bash
# Intel oneAPI compilers with Intel MPI
source env/env-oneapi2024-intelmpi-cpu.sh

# GCC with Intel MPI
source env/env-gcc-intelmpi-cpu.sh
```

For the GPU build:

```bash
source env/env-gcc73-hpcx-gpu.sh
```

The main environment variables used by the Makefile are:

| Variable | Description |
|---|---|
| `MPI_DIR` | MPI installation root |
| `FFTW_DIR` | Double-precision FFTW installation root |
| `MKL_ROOT` | Intel oneMKL installation root |
| `CUDA_DIR` | CUDA Toolkit root; required for GPU builds |
| `NVCC` | Path to `nvcc`; optional if it is under `CUDA_DIR/bin` |
| `CUDA_MATH_DIR` | Optional additional CUDA math-library root |

The example environment scripts contain site-specific paths and may
need to be modified for another cluster.

## Build instructions

Run all commands from the LibISDF project root.

### CPU build

```bash
source env/env-oneapi2024-intelmpi-cpu.sh

make clean
make USE_GPU=0 info
make USE_GPU=0 -j8
```

### GPU build

Build without CUDA-aware MPI:

```bash
source env/env-gcc73-hpcx-gpu.sh

make clean
make USE_GPU=1 USE_GPUDIRECT=0 CUDA_ARCH=80 info
make USE_GPU=1 USE_GPUDIRECT=0 CUDA_ARCH=80 -j8
```

Build with CUDA-aware MPI:

```bash
source env/env-gcc73-hpcx-gpu.sh

make clean
make USE_GPU=1 USE_GPUDIRECT=1 CUDA_ARCH=80 info
make USE_GPU=1 USE_GPUDIRECT=1 CUDA_ARCH=80 -j8
```

Common CUDA architecture values are:

| GPU | `CUDA_ARCH` |
|---|---:|
| NVIDIA V100 | `70` |
| NVIDIA A100 | `80` |
| NVIDIA H100 | `90` |

A successful build generates:

```text
libisdf.a
readpsi
```

Run `make clean` before switching compilers, MPI implementations,
CPU/GPU modes, GPUDirect settings, or CUDA architectures.

## Main Makefile options

| Variable | Values | Default | Description |
|---|---|---:|---|
| `COMPILE_MODE` | `release`, `debug` | `release` | Build mode |
| `USE_GPU` | `0`, `1` | `0` | Select CPU or GPU implementation |
| `USE_GPUDIRECT` | `0`, `1` | `0` | Enable the CUDA-aware MPI code path |
| `MPI_FLAVOR` | `auto`, `intel`, `generic` | `auto` | Select the MPI wrapper family |
| `MPI_IMPL` | `intelmpi`, `openmpi`, `mpich` | auto-detected | Select the matching MKL BLACS library |
| `FORTRAN_RT` | `ifcore`, `gfortran`, `none` | `none` | Additional Fortran runtime |
| `MKL_THREADING` | `sequential`, `intel_thread`, `gnu_thread` | `sequential` | MKL threading layer |
| `CUDA_ARCH` | `70`, `80`, `90`, ... | `80` | CUDA target architecture |
| `NPROCS` | positive integer | `2` | MPI processes used by `make run` |

Useful Makefile targets are:

```bash
make              # Build the library and the readpsi example
make lib          # Build libisdf.a only
make example      # Build the readpsi example
make info         # Print the resolved configuration
make clean        # Remove generated files
make run          # Run the bundled example
make help         # Print Makefile help
```

## Programming interface

Include the public header with:

```cpp
#include "isdf.hpp"
```

LibISDF exposes one `ISDF()` entry point. The CPU and GPU declarations
are selected at compile time and are not available simultaneously in
the same library build.

### CPU interface

```cpp
void ISDF(double *psirow,
          Domain domain_,
          int nocc,
          int nstate,
          int nv1,
          int nc1,
          int nv2,
          int nc2,
          int mu_points,
          double *thetaCol,
          int *piv,
          int KmeansMaxIter_ISDF,
          int scalblocksize = 64,
          std::string s = "scalapack",
          std::string s1 = "Kmeans",
          bool check = false);
```

### GPU interface

```cpp
void ISDF(double *d_psirow,
          Domain domain_,
          int nocc,
          int nstate,
          int nv1,
          int nc1,
          int nv2,
          int nc2,
          int mu_points,
          double *d_thetaCol,
          int *piv,
          int KmeansMaxIter_ISDF,
          bool check = false);
```

### Common arguments

| Argument | Type | I/O | Memory | Description |
|---|---|---|---|---|
| `psirow` / `d_psirow` | `double*` | Input | Host / device | Local block of the distributed wavefunction matrix |
| `domain_` | `Domain` | Input | Host | Real-space grids, simulation-cell information, and MPI communicators |
| `nocc` | `int` | Input | Scalar | Number of occupied states |
| `nstate` | `int` | Input | Scalar | Total number of Kohn-Sham states |
| `nv1`, `nc1` | `int` | Input | Scalar | Valence and conduction states in the first orbital subset |
| `nv2`, `nc2` | `int` | Input | Scalar | Valence and conduction states in the second orbital subset |
| `mu_points` | `int` | Input | Scalar | Number of interpolation points; `0` requests automatic selection |
| `thetaCol` / `d_thetaCol` | `double*` | Output | Host / device | Local block of the interpolation-vector matrix |
| `piv` | `int*` | Output | Host | Global grid-point permutation of length `Nr`; the first `Nmu` entries identify the selected interpolation points |
| `KmeansMaxIter_ISDF` | `int` | Input | Scalar | Maximum number of K-means iterations |
| `check` | `bool` | Input | Scalar | Enable additional numerical-error diagnostics |

The following arguments are available only in the CPU interface:

| Argument | Type | Default | Description |
|---|---|---:|---|
| `scalblocksize` | `int` | `64` | ScaLAPACK block size |
| `s` | `std::string` | `"scalapack"` | CPU backend: `"MPI"` or `"scalapack"` |
| `s1` | `std::string` | `"Kmeans"` | Point-selection method: `"QRCP"` or `"Kmeans"` |

In the CPU interface, `psirow` and `thetaCol` point to host memory. In
the GPU interface, `d_psirow` and `d_thetaCol` must point to memory on
the CUDA device assigned to the calling MPI process.

The caller is responsible for allocating and releasing all input and
output buffers.

## Distributed data layout

The global wavefunction matrix has dimensions

```text
Nr x Nb
```

where `Nr` is the number of real-space grid points and `Nb` is the
number of Kohn-Sham states.

The matrix uses a one-dimensional row-block distribution along the
real-space grid dimension. Each MPI process stores its local rows and
all state columns.

The global interpolation-vector matrix has dimensions

```text
Nr x Nmu
```

and is distributed over MPI processes along the interpolation-point
dimension using a one-dimensional column distribution.

`ISDF()` is collective over the MPI communicator stored in `domain_`.
All processes in that communicator must call the function with
consistent global parameters.

For GPU builds, each MPI process must select its CUDA device before
allocating device buffers and calling `ISDF()`.

## Si8 example

A two-process Si8 example is provided under:

```text
examples/si8/
```

After building LibISDF, run it from the project root:

```bash
mpirun -np 2 ./readpsi
```

See the [Si8 example README](examples/si8/README.md) for the system
parameters and input-data description.

## Slurm submission

Submit jobs from the project root.

CPU job:

```bash
sbatch scripts/run-cpu.slurm
```

GPU job:

```bash
sbatch scripts/run-gpu.slurm
```

The executable must be built with the corresponding CPU or GPU
environment before submission.

## Troubleshooting

Before switching build configurations, run:

```bash
make clean
```

Check shared-library dependencies with:

```bash
ldd ./readpsi | grep -E 'mpi|fftw|mkl|cuda|not found'
```

There should be no `not found` entries.

Common issues include:

- FFTW MPI was built against a different MPI implementation.
- `MPI_IMPL` does not match the selected MPI and MKL BLACS library.
- `CUDA_ARCH` does not match the target GPU.
- `USE_GPUDIRECT=1` is used with an MPI library without CUDA support.
- Multiple MPI processes are assigned to the same GPU.

