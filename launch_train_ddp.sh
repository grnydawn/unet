#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J Bias_ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
##SBATCH -o flash-%j.out
##SBATCH -e flash-%j.error

source /lustre/orion/cli115/proj-shared/grnydawn/repos/github/unet/venv/bin/activate

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

#eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

module load PrgEnv-gnu
module load gcc/12.2.0
#module load rocm/5.7.0 libtool
module load rocm/5.7.0

export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl/build:/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl-plugin-rocm570/lib/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-5.7.0/lib:$LD_LIBRARY_PATH

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/../src:$PYTHONPATH

#    model_options = {
#    "unet": UNet,
#    "residual_unet": ResidualUNet,
#    "attention_unet": AttentionUNet,
#    "unet_plus_plus": UNetPlusPlus,
#    "residual_unet_plus": ResidualUNetPlusPlus
#        }


#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
time srun --ntasks-per-node=8 -n $((SLURM_JOB_NUM_NODES*8)) \
	python Train_individual_ddp.py --base_channels 16 --batch_size 4 --model residual_unet_plus --dataset ResidualUNetPlusPlus
	#python Train_individual_ddp.py --base_channels 16 --batch_size 16 --model residual_unet_plus --dataset ResidualUNetPlusPlus


