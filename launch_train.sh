#
#
# pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.2.4

module load rocm/6.2.4
module load craype-accel-amd-gfx90a

export MIOPEN_USER_DB_PATH=/tmp/unet
export MIOPEN_DISABLE_CACHE=1

mkdir -p $MIOPEN_USER_DB_PATH
rm -rf $MIOPEN_USER_DB_PATH/*

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not active. Activating..."
    source /lustre/orion/cli115/proj-shared/grnydawn/repos/github/unet/venv/bin/activate
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

python Train.py \
	--base_channels 16 \
	--batch_size 8 \
	--model residual_unet_plus \
	--dataset ResidualUNetPlusPlus

	#-u \	# force the stdout and stderr streams to be unbuffered
