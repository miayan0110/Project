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
# data_path=./mit_multiview_test/

CUDA_VISIBLE_DEVICES="4, 7" python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=${port} main.py \
--gpu_id 4 \
--data_path ./datasets/miiw_train/train \
--save_per_epoch 10 \
--num_epochs 500 \