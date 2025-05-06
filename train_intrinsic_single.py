import argparse
import os
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
    parser.add_argument('--resize_size', default=64, type=int)   # resize size
    parser.add_argument('--resume', action='store_true')    # whether keep training the previous model or not

    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()
    return args


def main(args):
    pretrained_model = load_latent_intrinsic(args.latent_intrinsic_weight, args.gpu_id)
    
    dataset = LSUNDataset(pretrained_model, args)
    # dataset = StyLitGAN_Dataset(pretrained_model, args)
    # dataset = MIIWDataset(pretrained_model, args)
    diff_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train_intrinsic_diffusion(diff_dataloader, args, device=f'cuda:{args.gpu_id}', save_root=args.intrinsic_ckpt_root, resume=args.resume)
    # train_img_diffusion(diff_dataloader, args, device=f'cuda:{args.gpu_id}', save_root='./ckpt/image_diff', resume=args.resume)
    # save_latent(diff_dataloader, './datasets/lsun_bedroom/lsun_train_code')

    # eval(args, device=f'cuda:{args.gpu_id}')    
    # eval(args, pretrained_model, device=f'cuda:{args.gpu_id}')    


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.intrinsic_ckpt_root, exist_ok=True)
    os.makedirs(args.extrinsic_ckpt_root, exist_ok=True)
    os.makedirs(args.decoder_ckpt_root, exist_ok=True)
    main(args)