#!/bin/bash
#SBATCH -A nwp500
#SBATCH -J Bias_ddp
#SBATCH --nodes=4
##SBATCH --ntasks-per-node=8
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -q debug
##SBATCH -o flash-%j.out
##SBATCH -e flash-%j.error

source /lustre/orion/cli115/proj-shared/grnydawn/repos/github/unet/venv/bin/activate

module reset
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export MIOPEN_DISABLE_CACHE=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=hsn
export NCCL_PROTO=Simple
export NCCL_IB_DISABLE=1
#export NCCL_DEBUG=INFO
NUM_NODES=4
NUM_TASKS=32

time srun -N ${NUM_NODES} --ntasks-per-node=8 -n ${NUM_TASKS} \
	python Train_individual_ddp.py \
		--base_channels 16 \
		--batch_size 4 \
		--model residual_unet_plus \
		--dataset ResidualUNetPlusPlus \
		--outdir "/lustre/orion/cli115/scratch/grnydawn/unet_${NUM_NODES}"

