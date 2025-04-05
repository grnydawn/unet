#!/bin/bash
#SBATCH -A lrn036
#SBATCH -J B_cor_inf
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
##SBATCH --gpus-per-node=8
##SBATCH --gpus-per-task=1
##SBATCH --ntasks-per-node=8
##SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o train-%j.out
#SBATCH -e train-%j.out

#source ~/miniconda_frontier/etc/profile.d/conda.sh
source /lustre/orion/cli115/proj-shared/grnydawn/repos/github/unet/venv/bin/activate

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

#eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0 libtool

export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl/build:/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl-plugin-rocm570/lib/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-5.7.0/lib:$LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

#    model_options = {
#    "unet": UNet,
#    "residual_unet": ResidualUNet,
#    "attention_unet": AttentionUNet,
#    "unet_plus_plus": UNetPlusPlus,
#    "residual_unet_plus": ResidualUNetPlusPlus
#        }

python -u Train.py --base_channels 16 --batch_size 8 --model residual_unet_plus --dataset ResidualUNetPlusPlus

# torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.23 GiB. GPU 0 has a total capacity of 63.98 GiB of which 1.88 GiB is free. Of the allocated memory 60.87 GiB is allocated by PyTorch, and 941.24 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_HIP_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
#python -u Train.py --base_channels 16 --batch_size 32 --model residual_unet_plus --dataset ResidualUNetPlusPlus
#python -u Train.py
#time srun python -m neural_lam.train_model --model graph_lam  --epochs 2
#time srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gres=gpu:8 --ntasks-per-node=8 \
#time srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c1 --gres=gpu:8 --ntasks-per-node=8 \

#export HSA_OVERRIDE_GFX_VERSION=11.0.0
#time srun python Train_individual_ddp.py
#time srun -n $((SLURM_JOB_NUM_NODES*8)) python Train_individual_ddp.py
