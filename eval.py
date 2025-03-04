import argparse
import os
from torch.utils.data import DataLoader
from utils import *
from diffusions import *
from diff import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_intrinsic_weight', default='./pretrained_weight/latent_intrinsic.pth.tar', type=str)  # latent intrinsic weight path
    parser.add_argument('--gpu_id', default=7, type=int)   # id of usage gpu
    parser.add_argument('--data_path', default='./datasets/miiw_test/test', type=str)    # dataset path
    parser.add_argument('--intrinsic_ckpt_root', default='./ckpt/intrinsic', type=str)    # intrinsic checkpoint save path
    parser.add_argument('--extrinsic_ckpt_root', default='./ckpt/extrinsic', type=str)    # extrinsic checkpoint save path
    parser.add_argument('--decoder_ckpt_root', default='./ckpt/decoder', type=str)    # decoder checkpoint save path
    parser.add_argument('--save_per_epoch', default=10, type=int)    # save checkpoint per epoch

    parser.add_argument('--lr', default=1e-4, type=float)   # learning rate
    parser.add_argument('--batch_size', default=4, type=int)   # batch size
    parser.add_argument('--num_epochs', default=10, type=int)   # training number of epochs
    parser.add_argument('--num_train_timesteps', default=1000, type=int)   # training number of timesteps
    parser.add_argument('--resume', action='store_true')    # whether keep training the previous model or not

    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()
    return args


def main(args):
    # dataset = StyLitGAN_Dataset(args.data_path)
    pretrained_model = load_latent_intrinsic(args.latent_intrinsic_weight, args.gpu_id)
    diff_dataloader = DataLoader(MIIWDataset(pretrained_model, args), batch_size=args.batch_size, shuffle=True)
    decoder_dataloader = DataLoader(MIIWDataset(pretrained_model, args, for_decoder=True), batch_size=args.batch_size, shuffle=True)


    train_intrinsic_diffusion(diff_dataloader, args, device=f'cuda:{args.gpu_id}', save_root=args.intrinsic_ckpt_root, resume=False)
    train_extrinsic_diffusion(diff_dataloader, args, device=f'cuda:{args.gpu_id}', save_root=args.extrinsic_ckpt_root, resume=False)
    train_decoder(decoder_dataloader, args, device=f'cuda:{args.gpu_id}', save_root=args.decoder_ckpt_root, resume=False)
    


if __name__ == '__main__':
    args = get_args()
    os.makedirs('./visualize', exist_ok=True)
    main(args)