
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
--nproc_per_node=1 --master_port=${port} eval.py \
--gpu_id 5 \
--data_path ./datasets/lsun_bedroom/lsun_train \
--num_train_timesteps 250 \
--resize_size 256 \
--eval_mode test_dis \
--eval_result_save_root 226_in \

# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7