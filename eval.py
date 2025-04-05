import argparse
import os
import glob
from torch.utils.data import DataLoader

from models import *
from model_utils import *
from dataset import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_intrinsic_weight', default='./pretrained_weight/latent_intrinsic.pth.tar', type=str)  # latent intrinsic weight path
    parser.add_argument('--gpu_id', default=7, type=int)   # id of usage gpu
    parser.add_argument('--data_path', default='./datasets/miiw_train/train', type=str)    # dataset path
    parser.add_argument('--intrinsic_ckpt_root', default='./ckpt/intrinsic', type=str)    # intrinsic checkpoint save path
    parser.add_argument('--extrinsic_ckpt_root', default='./ckpt/extrinsic', type=str)    # extrinsic checkpoint save path
    parser.add_argument('--decoder_ckpt_root', default='./ckpt/decoder', type=str)    # decoder checkpoint save path
    parser.add_argument('--save_per_epoch', default=5, type=int)    # save checkpoint per epoch

    parser.add_argument('--lr', default=1e-4, type=float)   # learning rate
    parser.add_argument('--batch_size', default=8, type=int)   # batch size
    parser.add_argument('--num_epochs', default=50, type=int)   # training number of epochs
    parser.add_argument('--num_train_timesteps', default=1000, type=int)   # training number of timesteps
    parser.add_argument('--resize_size', default=256, type=int)   # resize size
    parser.add_argument('--resume', action='store_true')    # whether keep training the previous model or not
    parser.add_argument('--eval_mode', default='all', type=str)    # which model to evaluate ('all', 'intrinsic', 'extrinsic', 'decoder')

    parser.add_argument('--eval_result_save_root', default='0', type=str)    # which model to evaluate ('all', 'intrinsic', 'extrinsic', 'decoder')

    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()
    return args


def main(args):
    intrinsic_list = glob.glob(f'{args.intrinsic_ckpt_root}/*.pth')
    intrinsic_list.sort()
    args.intrinsic_path = intrinsic_list[-1]
    args.intrinsic_path = './ckpt/intrinsic/checkpoint_050.pth'

    extrinsic_list = glob.glob(f'{args.extrinsic_ckpt_root}/*.pth')
    extrinsic_list.sort()
    args.extrinsic_path = extrinsic_list[-1]
    args.extrinsic_path = './ckpt/extrinsic/checkpoint_050.pth'

    decoder_list = glob.glob(f'{args.decoder_ckpt_root}/*.pth')
    decoder_list.sort()
    args.decoder_path = decoder_list[-1]
    # args.decoder_path = './ckpt/decoder/checkpoint_024.pth'

    if args.eval_mode == 'all':
        args.eval_result_save_root = f'./eval_result/all/{args.eval_result_save_root}'
        os.makedirs(args.eval_result_save_root, exist_ok=True)
        eval(args, device=f'cuda:{args.gpu_id}')
    else:
        args.eval_result_save_root = f'./eval_result/part/{args.eval_result_save_root}'
        os.makedirs(args.eval_result_save_root, exist_ok=True)
        pretrained_model = load_latent_intrinsic(args.latent_intrinsic_weight, args.gpu_id)
        part_eval(args, pretrained_model, device=f'cuda:{args.gpu_id}')    


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.intrinsic_ckpt_root, exist_ok=True)
    os.makedirs(args.extrinsic_ckpt_root, exist_ok=True)
    os.makedirs(args.decoder_ckpt_root, exist_ok=True)
    main(args)