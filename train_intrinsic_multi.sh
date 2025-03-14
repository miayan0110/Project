
while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

NUM_GPUS=$(nvidia-smi -L | wc -l)  # 計算 GPU 數量
export WORLD_SIZE=$NUM_GPUS

python -m torch.distributed.launch \
--nproc_per_node=$NUM_GPUS --master_port=${port} train_intrinsic.py \
--data_path ./datasets/miiw_train/train \
--save_per_epoch 5 \
--batch_size 8 \
--num_epochs 50 \
--resume \

# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7