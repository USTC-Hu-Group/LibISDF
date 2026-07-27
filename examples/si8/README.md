# Si8 LibISDF Example

This directory contains a small Si8 example for testing the CPU and GPU
implementations of LibISDF. The example reads a distributed real-valued real-space
wavefunction matrix and calls the `ISDF()` interface.

## Directory structure

```text
examples/si8/
├── README.md
├── readpsi.cpp
└── data/
    ├── psirow0
    └── psirow1
```

## Input data

The global wavefunction matrix is distributed over two MPI processes
using a one-dimensional row-block distribution along the real-space
grid dimension. Each process reads its local block from:

```text
data/psirow<RANK>
```

For the bundled input:

```text
rank 0 -> data/psirow0
rank 1 -> data/psirow1
```

Therefore, this example must be run with exactly two MPI processes.

## System parameters

| Parameter | Value |
|---|---:|
| System | Si8 |
| Simulation-cell dimensions (bohr) | `10.216 x 10.216 x 10.216` |
| Cell origin | `(0.0, 0.0, 0.0)` |
| Coarse real-space grid | `16 x 16 x 16` |
| Fine real-space grid | `32 x 32 x 32` |
| Coarse-grid points | `4096` |
| Number of Kohn-Sham states | `35` |
| Number of occupied states | `16` |
| `Nv1` | `16` |
| `Nc1` | `0` |
| `Nv2` | `16` |
| `Nc2` | `0` |
| Number of ISDF interpolation points | `192` |
| Required MPI processes | `2` |

The length unit is the same as that used to generate the input
wavefunctions.

## Build

Build the desired CPU or GPU implementation from the LibISDF project
root. See the main [build instructions](../../README.md#build-instructions)
for details.

## Run

After building, run the example from the project root:


```bash
mpirun -np 2 ./readpsi
```

CPU and GPU jobs may also be submitted through Slurm:

```bash
sbatch scripts/run-cpu.slurm
sbatch scripts/run-gpu.slurm
```

The GPU job uses two MPI processes and two GPUs, with one GPU assigned
to each MPI process according to its node-local rank.


