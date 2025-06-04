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

#export MIOPEN_USER_DB_PATH=/tmp/$JOBID
#export MIOPEN_USER_DB_PATH=${MEMBERWORK}/cli190/tmp/miopen_cache
#mkdir -p $MIOPEN_USER_DB_PATH
#export MIOPEN_DISABLE_CACHE=1
#export MIOPEN_DISABLE_CACHE=0

#echo "Cache dir: ${MIOPEN_USER_DB_PATH}"

MASTER_ADDR1=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
MASTER_ADDR2=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '2p')
MASTER_ADDR3=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '3p')
MASTER_ADDR4=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '4p')

export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=hsn
export NCCL_PROTO=Simple
export NCCL_IB_DISABLE=1
#export NCCL_DEBUG=INFO
NUM_NODES=4
#NUM_TASKS=32
NUM_TRIES=6

OUTDIR="/lustre/orion/cli115/scratch/grnydawn/unet_leadtime.${NUM_NODES}.${NUM_TRIES}"

pairs=(${MASTER_ADDR1} 6 ${MASTER_ADDR2} 12 ${MASTER_ADDR3} 18 ${MASTER_ADDR4} 24)

for ((i=0; i<${#pairs[@]}; i+=2)); do
    MASTER_ADDR=${pairs[i]}
    LEAD_TIME=${pairs[i+1]}

    mkdir -p ${OUTDIR}/${MASTER_ADDR}/energy_start
    cp -rf /sys/cray/pm_counters/* $OUTDIR/${MASTER_ADDR}/energy_start

	MASTER_ADDR=${MASTER_ADDR} LEAD_TIME=${LEAD_TIME} time srun -N 1 --ntasks-per-node=8 -n 8 \
	python Train_individual_ddp_leadtime1.py \
		--leadtime ${LEAD_TIME} \
		--base_channels 16 \
		--batch_size 4 \
		--model residual_unet_plus \
		--dataset ResidualUNetPlusPlus \
		--outdir ${OUTDIR} &
done

#MASTER_ADDR=${MASTER_ADDR1} LEAD_TIME=6 time srun -N 1 --ntasks-per-node=8 -n 8 \
#	python Train_individual_ddp_leadtime1.py \
#		--leadtime 6 \
#		--base_channels 16 \
#		--batch_size 4 \
#		--model residual_unet_plus \
#		--dataset ResidualUNetPlusPlus \
#		--outdir "/lustre/orion/cli115/scratch/grnydawn/unet_leadtime.${NUM_NODES}.${NUM_TRIES}" &
#
#MASTER_ADDR=${MASTER_ADDR2} LEAD_TIME=12 time srun -N 1 --ntasks-per-node=8 -n 8 \
#	python Train_individual_ddp_leadtime1.py \
#		--leadtime 12 \
#		--base_channels 16 \
#		--batch_size 4 \
#		--model residual_unet_plus \
#		--dataset ResidualUNetPlusPlus \
#		--outdir "/lustre/orion/cli115/scratch/grnydawn/unet_leadtime.${NUM_NODES}.${NUM_TRIES}" &
#
#MASTER_ADDR=${MASTER_ADDR3} LEAD_TIME=18 time srun -N 1 --ntasks-per-node=8 -n 8 \
#	python Train_individual_ddp_leadtime1.py \
#		--leadtime 18 \
#		--base_channels 16 \
#		--batch_size 4 \
#		--model residual_unet_plus \
#		--dataset ResidualUNetPlusPlus \
#		--outdir "/lustre/orion/cli115/scratch/grnydawn/unet_leadtime.${NUM_NODES}.${NUM_TRIES}" &
#
#MASTER_ADDR=${MASTER_ADDR4} LEAD_TIME=24 time srun -N 1 --ntasks-per-node=8 -n 8 \
#	python Train_individual_ddp_leadtime1.py \
#		--leadtime 24 \
#		--base_channels 16 \
#		--batch_size 4 \
#		--model residual_unet_plus \
#		--dataset ResidualUNetPlusPlus \
#		--outdir "/lustre/orion/cli115/scratch/grnydawn/unet_leadtime.${NUM_NODES}.${NUM_TRIES}" &

wait

for ((i=0; i<${#pairs[@]}; i+=2)); do
    MASTER_ADDR=${pairs[i]}

    mkdir -p ${OUTDIR}/${MASTER_ADDR}/energy_stop
    cp -rf /sys/cray/pm_counters/* $OUTDIR/${MASTER_ADDR}/energy_stop

done
