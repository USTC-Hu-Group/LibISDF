#!/bin/bash
#SBATCH --job-name=testISDF
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --output=isdf_%j.out
#SBATCH --error=isdf_%j.err


cd $SLURM_SUBMIT_DIR


source env/env-oneapi2024-intelmpi-cpu.sh


echo "===== Job Info ====="
echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "JobID:   $SLURM_JOB_ID"
echo "Tasks:   $SLURM_NTASKS"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo ""
echo "===== Module list ====="
module list 2>&1
echo ""
echo "===== Binary Check ====="
ldd ./readpsi 2>&1 | grep -E '(not found|cuda|mpi|mkl|fftw)' | head -20
echo ""
echo "===== Run ====="
mpirun -np $SLURM_NTASKS ./readpsi

echo ""
echo "===== Done at $(date) ====="
