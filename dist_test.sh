CONFIG=$1
WORK_DIR_CKPT=$2
PORT=${PORT:-29500}

CUDA_VISIBLE_DEVICES="0" python test.py --config $CONFIG --checkpoint $WORK_DIR_CKPT --eval=mIoU 
