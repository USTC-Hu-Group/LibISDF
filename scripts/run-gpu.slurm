#!/bin/bash
#SBATCH --job-name=testISDF
#SBATCH --output=isdf-%j.out
#SBATCH --error=isdf-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
# #SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00


source env/env-gcc73-hpcx-gpu.sh
export UCX_TLS=sm,self,cuda_copy,cuda_ipc
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores


echo "=========== Job Info ==========="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node list    : $SLURM_NODELIST"
echo "Tasks total  : $SLURM_NTASKS"
echo "Tasks/node   : $SLURM_NTASKS_PER_NODE"
echo "CPUs/task    : $SLURM_CPUS_PER_TASK"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader
echo "================================"


mpirun -n "$SLURM_NTASKS" \
    --mca pml ucx \
    --mca coll_hcoll_enable 0 \
    --mca coll ^hcoll,ucc \
    ./readpsi
#mpirun  -n $SLURM_NTASKS ./readpsi

