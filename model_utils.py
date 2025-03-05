import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from diffusers import DDPMScheduler
from tqdm import tqdm
from PIL import Image
import numpy as np

from models import *
from unets import UNet

#----------------------------------------------------------------------------
# Training
#----------------------------------------------------------------------------

def train_intrinsic_diffusion(train_loader, args, device="cuda", save_root="./ckpt/intrinsic", resume=False):
    model = Diffusion(sample_size=128, in_channels=110).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    criterion = nn.MSELoss()
    start_epoch = 0
    
    if resume:
        ckpt_list = glob.glob(f'{save_root}/*.pth')
        ckpt_list = ckpt_list.sort()
        model, optimizer, start_epoch, _ = load_model(model, optimizer, ckpt_list[-1], device)
    
    model.train()
    print('\nstart training intrinsic diffusion...')
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in pbar:
            intrinsic, _ = batch   
            inputs = intrinsic.squeeze(1).to(device) # shape: [batch, 110, 128, 128]
            optimizer.zero_grad()
            
            noise = torch.randn_like(inputs)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (inputs.shape[0],), device=device)

            noisy_inputs = noise_scheduler.add_noise(inputs, noise, timesteps)
            
            # print(f'noisy_inputs.shape: {noisy_inputs.shape}, timesteps.shape: {timesteps.shape}')
            outputs = model(noisy_inputs, timesteps).sample
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.6f}")
        if (epoch+1) % args.save_per_epoch == 0:
            save_model(model, optimizer, epoch + 1, epoch_loss / len(train_loader), f'{save_root}/checkpoint_{epoch+1}.pth')


def train_extrinsic_diffusion(train_loader, args, device="cuda", save_root="./ckpt/extrinsic", resume=False):
    model = Diffusion(sample_size=128, in_channels=64, is_intrinsic=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    criterion = nn.MSELoss()
    start_epoch = 0
    
    if resume:
        ckpt_list = glob.glob(f'{save_root}/*.pth')
        ckpt_list = ckpt_list.sort()
        model, optimizer, start_epoch, _ = load_model(model, optimizer, ckpt_list[-1], device)
    
    model.train()
    print('\nstart training extrinsic diffusion...')
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in pbar:
            # tensor shape: [batch, 16]
            _, extrinsic = batch
            inputs = extrinsic.to(device)
            inputs = inputs.view(inputs.shape[0], -1, 1, 1)
            optimizer.zero_grad()
            
            noise = torch.randn_like(inputs)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (inputs.shape[0],), device=device)
            noisy_inputs = noise_scheduler.add_noise(inputs, noise, timesteps)
            
            # print(f'noisy_inputs.shape: {noisy_inputs.shape}, timesteps.shape: {timesteps.shape}')
            outputs = model(noisy_inputs, timesteps)
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.6f}")
        if (epoch+1) % args.save_per_epoch == 0:
            save_model(model, optimizer, epoch + 1, epoch_loss / len(train_loader), f'{save_root}/checkpoint_{epoch+1}.pth')

def train_decoder(train_loader, args, device="cuda", save_root="./ckpt/decoder", resume=False):
    decoder = Decoder(in_channels=3, out_channels=3, latent_channels=174).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    start_epoch = 0

    if resume:
        ckpt_list = glob.glob(f'{save_root}/*.pth')
        ckpt_list = ckpt_list.sort()
        decoder, optimizer, start_epoch, _ = load_model(decoder, optimizer, ckpt_list[-1], device)

    decoder.train()
    print('\nstart training decoder...')
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch in pbar:
            target_img, intrinsic, extrinsic = batch
            target_img = target_img.squeeze(1).to(device)          # target_img.shape: [batch, C, H, W]
            intrinsic = intrinsic.squeeze(1).to(device) # intrinsic.shape: [batch, 110, 128, 128]
            extrinsic = extrinsic.to(device)            # extrinsic.shape: [batch, 16]

            optimizer.zero_grad()
            
            # Pass latent representation through decoder
            reconstructed_img = decoder(intrinsic, extrinsic)  # Output shape: [batch, C, H, W]

            # Compute loss
            loss = criterion(reconstructed_img, target_img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.6f}")
        if (epoch+1) % args.save_per_epoch == 0:
            save_model(decoder, optimizer, epoch + 1, epoch_loss / len(train_loader), f'{save_root}/checkpoint_{epoch+1}.pth')
            save_visualize_images(reconstructed_img, target_img)

#----------------------------------------------------------------------------
# Evaluation
#----------------------------------------------------------------------------

def eval(args, pretrained_model, device="cuda"):
    print('\nstart evalution...')
    torch.cuda.empty_cache()

    with torch.inference_mode():
        # img_transform = transforms.Compose([
        #     torchvision.transforms.Resize(256),
        #     torchvision.transforms.CenterCrop((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        #             ),
        # ])
        # img1 = Image.open('./datasets/miiw_train/14n_copyroom1/dir_2_mip2.jpg').convert("RGB")
        # img2 = Image.open('./datasets/miiw_train/summer_kitchen1/dir_22_mip2.jpg').convert("RGB")
        
        # img1 = img_transform(img1).unsqueeze(0).to(device)
        # img2 = img_transform(img2).unsqueeze(0).to(device)

        # intrinsic, _ = get_image_intrinsic_extrinsic(pretrained_model, img1)
        # _, extrinsic_latent = get_image_intrinsic_extrinsic(pretrained_model, img2)
        # flatten_intrinsic = [tensor.flatten(start_dim=1) for tensor in intrinsic]
        # intrinsic_latent = torch.cat(flatten_intrinsic, dim=1).view(intrinsic[0].shape[0], -1, 128, 128)
        # extrinsic_latent = extrinsic_latent.view(extrinsic_latent.shape[0], -1, 1, 1)

        # intrinsic diffusion
        intrinsic_diff = Diffusion(sample_size=128, in_channels=110).to(device)
        intrinsic_optimizer = optim.AdamW(intrinsic_diff.parameters(), lr=args.lr)
        intrinsic_diff, _, _, _ = load_model(intrinsic_diff, intrinsic_optimizer, save_path=f'{args.intrinsic_ckpt_root}/checkpoint_1.pth', device=device)

        intrinsic_diff.eval()
        intrinsic_latent = intrinsic_diff(torch.randn(1, 110, 128, 128).to(device), timestep=1000).sample  # [1, 110, 128, 128]
        del intrinsic_diff
        torch.cuda.empty_cache()

        # extrinsic diffusion
        extrinsic_diff = Diffusion(sample_size=128, in_channels=64, is_intrinsic=False).to(device)
        extrinsic_optimizer = optim.AdamW(extrinsic_diff.parameters(), lr=args.lr)
        extrinsic_diff, _, _, _ = load_model(extrinsic_diff, extrinsic_optimizer, save_path=f'{args.extrinsic_ckpt_root}/checkpoint_1.pth', device=device)
        
        extrinsic_diff.eval()
        extrinsic_latent = extrinsic_diff(torch.randn(1, 16, 1, 1).to(device), timestep=1000)  # [1, 16, 1, 1]
        del extrinsic_diff
        torch.cuda.empty_cache()

        # decoder
        decoder = Decoder(in_channels=3, out_channels=3, latent_channels=174).to(device)
        decoder_optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
        decoder, _, _, _ = load_model(decoder, decoder_optimizer, save_path=f'{args.decoder_ckpt_root}/checkpoint_1.pth', device=device)

        decoder.eval()
        output_image = decoder(intrinsic_latent, extrinsic_latent.squeeze(2).squeeze(2))  # [1, 3, 256, 256]

        save_result_images(output_image, "generated_image.png")

#----------------------------------------------------------------------------
# Utils
#----------------------------------------------------------------------------

def get_image_intrinsic_extrinsic(model, img):
    model.eval()

    P_mean=-0.5
    P_std=1.2
    rnd_normal = torch.randn([img.shape[0], 1, 1, 1], device=img.device)
    sigma = (rnd_normal * P_std + P_mean).exp().to(img.device) * 0 + 0.001

    noise = torch.randn_like(img)
    noisy_img = img + noise * sigma

    intrinsic, extrinsic = model(img, run_encoder = True)
    return intrinsic, extrinsic

def load_latent_intrinsic(path, device):
    # initialize
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

    model = UNet(img_resolution = 256, in_channels = 3, out_channels = 3,
                     num_blocks_list = [1, 2, 2, 4, 4, 4], attn_resolutions = [0], model_channels = 32,
                     channel_mult = [1, 2, 4, 4, 8, 16], affine_scale = float(5e-3))
    model.cuda(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)

    # load pretrained latent intrinsic model
    print(f"=> loading checkpoint '{path}'...")
    checkpoint = torch.load(path, map_location=f'cuda:{device}')
    model.load_state_dict(checkpoint['state_dict'])
    print('=> finished.')

    return model

def load_model(model, optimizer, save_path="diffusion_checkpoint.pth", device="cuda"):
    print(f"=> loading checkpoint '{save_path}'...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('=> finished.')

    return model, optimizer, epoch, loss

def save_model(model, optimizer, epoch, loss, save_path="diffusion_checkpoint.pth"):
    print(f"=> saving checkpoint to '{save_path}'...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print('=> finished.')