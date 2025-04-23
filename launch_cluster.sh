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

module reset
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

#module load amd-mixed/5.7.0
#module load craype-accel-amd-gfx90a

#module load PrgEnv-gnu
#module load rocm/6.2.4
#module unload darshan-runtime
#module unload libfabric
#module load craype-accel-amd-gfx90a

#module load rocm
#eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

#module load PrgEnv-gnu
#module load gcc/12.2.0
##module load rocm/5.7.0 libtool
#module load rocm/5.7.0

#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl/build:/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl-plugin-rocm570/lib/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-5.7.0/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl/build:/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl-plugin-rocm570/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/ccs/home/grnydawn/prjfrontier/opts/rccl/lib:${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

echo "JOBID = $JOBID"
echo "JOBSIZE = $JOBSIZE"

#export MIOPEN_DISABLE_CACHE=1
#export NCCL_PROTO=Simple
#export MIOPEN_USER_DB_PATH=/tmp/$JOBID
export MIOPEN_USER_DB_PATH=/lustre/orion/cli115/scratch/grnydawn/temp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
#export PYTHONNOUSERSITE=1
#export OMP_NUM_THREADS=7
#export PYTHONPATH=$PWD/../src:$PYTHONPATH
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,ens2,bond0
#export NCCL_SOCKET_IFNAME=hsn0
#export NCCL_DEBUG=info

for NUM_NODES in 1
do

NTASKS_PER_NODE=4

#NUM_TASKS=$((NUM_NODES*8))
NUM_TASKS=$((NUM_NODES*NTASKS_PER_NODE))

echo "NUM_NODES = $NUM_NODES"
echo "NUM_TASKS = $NUM_TASKS"

#time srun -N ${NUM_NODES} --ntasks-per-node=${NTASKS_PER_NODE} -n ${NUM_TASKS} \
time srun -N ${NUM_NODES} -n ${NUM_TASKS} \
	python cluster_ddp2.py

done
