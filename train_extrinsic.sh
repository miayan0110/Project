
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

python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=${port} train_extrinsic.py \
--gpu_id 1 \
--data_path ./datasets/miiw_train/train \
--save_per_epoch 5 \
--batch_size 8 \
--num_epochs 50 \
--resume \
