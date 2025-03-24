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

from torch.utils.tensorboard import SummaryWriter

from models import *
from unets import UNet

#----------------------------------------------------------------------------
# Tensorboard
#----------------------------------------------------------------------------

logdir = './records'
os.makedirs(logdir, exist_ok=True)

#----------------------------------------------------------------------------
# Training
#----------------------------------------------------------------------------

def train_intrinsic_diffusion(train_loader, args, device="cuda", save_root="./ckpt/intrinsic", resume=False):
    log_subdir = f'{logdir}/intrinsic_loss'
    os.makedirs(log_subdir, exist_ok=True)
    writer = SummaryWriter(log_subdir)

    model = Diffusion(sample_size=128, in_channels=110).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    criterion = nn.MSELoss()
    start_epoch = 0
    
    if resume:
        ckpt_list = glob.glob(f'{save_root}/*.pth')
        ckpt_list.sort()
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
            
            pred_noise = model(noisy_inputs, timesteps).sample
            loss = criterion(pred_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.6f}")
        writer.add_scalar('Intrinsic Loss', epoch_loss / len(train_loader), epoch+1)
        if (epoch+1) % args.save_per_epoch == 0:
            save_model(model, optimizer, epoch + 1, epoch_loss / len(train_loader), f'{save_root}/checkpoint_{epoch+1:03d}.pth')


def train_extrinsic_diffusion(train_loader, args, device="cuda", save_root="./ckpt/extrinsic", resume=False):
    log_subdir = f'{logdir}/extrinsic_loss'
    os.makedirs(log_subdir, exist_ok=True)
    writer = SummaryWriter(log_subdir)

    model = Diffusion(sample_size=128, in_channels=64, is_intrinsic=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    criterion = nn.MSELoss()
    start_epoch = 0
    
    if resume:
        ckpt_list = glob.glob(f'{save_root}/*.pth')
        ckpt_list.sort()
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
            
            pred_noise = model(noisy_inputs, timesteps)
            loss = criterion(pred_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.6f}")
        writer.add_scalar('Extrinsic Loss', epoch_loss / len(train_loader), epoch+1)
        if (epoch+1) % args.save_per_epoch == 0:
            save_model(model, optimizer, epoch + 1, epoch_loss / len(train_loader), f'{save_root}/checkpoint_{epoch+1:03d}.pth')

def train_decoder(train_loader, args, device="cuda", save_root="./ckpt/decoder", resume=False):
    log_subdir = f'{logdir}/decoder_loss'
    os.makedirs(log_subdir, exist_ok=True)
    writer = SummaryWriter(log_subdir)

    decoder = Decoder(in_channels=3, out_channels=3, latent_channels=174).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    start_epoch = 0

    if resume:
        ckpt_list = glob.glob(f'{save_root}/*.pth')
        ckpt_list.sort()
        decoder, optimizer, start_epoch, _ = load_model(decoder, optimizer, ckpt_list[-1], device)

    decoder.train()
    print('\nstart training decoder...')
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch in pbar:
            scene_img, light_img, scene_intrinsic, scene_extrinsic, light_extrinsic = batch
            scene_img = scene_img.squeeze(1).to(device)          # scene_img.shape: [batch, C, H, W]
            light_img = light_img.squeeze(1).to(device)          # light_img.shape: [batch, C, H, W]
            scene_intrinsic = scene_intrinsic.squeeze(1).to(device) # intrinsic.shape: [batch, 110, 128, 128]
            scene_extrinsic = scene_extrinsic.to(device)            # extrinsic.shape: [batch, 16]
            scene_extrinsic = scene_extrinsic.view(scene_extrinsic.shape[0], -1, 1, 1)    # extrinsic.shape: [batch, 16, 1, 1]
            light_extrinsic = light_extrinsic.to(device)            # extrinsic.shape: [batch, 16]
            light_extrinsic = light_extrinsic.view(light_extrinsic.shape[0], -1, 1, 1)    # extrinsic.shape: [batch, 16, 1, 1]

            optimizer.zero_grad()
            
            # Pass latent representation through decoder
            reconstructed_img = decoder(scene_intrinsic, scene_extrinsic)  # Output shape: [batch, C, H, W]
            relighted_img = decoder(scene_intrinsic, light_extrinsic)  # Output shape: [batch, C, H, W]

            # Compute loss: reconstruct loss + relight loss
            loss = criterion(reconstructed_img, scene_img) + criterion(relighted_img, light_img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.6f}")
        writer.add_scalar('Decoder Loss', epoch_loss / len(train_loader), epoch+1)
        if (epoch+1) % args.save_per_epoch == 0:
            save_model(decoder, optimizer, epoch + 1, epoch_loss / len(train_loader), f'{save_root}/checkpoint_{epoch+1:03d}.pth')


#----------------------------------------------------------------------------
# Evaluation
#----------------------------------------------------------------------------

def part_eval(args, pretrained_model, device="cuda"):
    print('\nstart part evalution...')
    torch.cuda.empty_cache()

    with torch.inference_mode():
        img_transform = transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        in_img = Image.open('./datasets/my_data/ori_1.jpg').convert("RGB")
        ex_img = Image.open('./datasets/my_data/ref_1.jpg').convert("RGB")
        # in_img = Image.open('./datasets/stylitgan_train/0_0.jpg').convert("RGB")
        # ex_img = Image.open('./datasets/stylitgan_train/0_7.jpg').convert("RGB")
        
        in_img = img_transform(in_img).unsqueeze(0).to(device)
        ex_img = img_transform(ex_img).unsqueeze(0).to(device)

        intrinsic_latent, _ = get_image_intrinsic_extrinsic(pretrained_model, in_img)

        _, extrinsic_latent = get_image_intrinsic_extrinsic(pretrained_model, ex_img)
        extrinsic_latent = extrinsic_latent.view(extrinsic_latent.shape[0], -1, 1, 1)

        if args.eval_mode == 'intrinsic':
            print('evaluate intrinsic diff...')
            save_result_images(ex_img, f'{args.eval_result_save_root}/light.jpg')
            intrinsic_diff = Diffusion(sample_size=128, in_channels=110).to(device)
            intrinsic_optimizer = optim.AdamW(intrinsic_diff.parameters(), lr=args.lr)
            intrinsic_diff, _, _, _ = load_model(intrinsic_diff, intrinsic_optimizer, save_path=args.intrinsic_path, device=device)

            intrinsic_diff.eval()
            intrinsic_latent = intrinsic_diff(torch.randn(1, 110, 128, 128).to(device), timestep=1000).sample  # [1, 110, 128, 128]
            del intrinsic_diff
            torch.cuda.empty_cache()
        elif args.eval_mode == 'extrinsic':
            print('evaluate extrinsic diff...')
            save_result_images(in_img, f'{args.eval_result_save_root}/scene.jpg')
            extrinsic_diff = Diffusion(sample_size=128, in_channels=64, is_intrinsic=False).to(device)
            extrinsic_optimizer = optim.AdamW(extrinsic_diff.parameters(), lr=args.lr)
            extrinsic_diff, _, _, _ = load_model(extrinsic_diff, extrinsic_optimizer, save_path=args.extrinsic_path, device=device)
            
            extrinsic_diff.eval()
            extrinsic_latent = extrinsic_diff(torch.randn(1, 16, 1, 1).to(device), timestep=1000)  # [1, 16, 1, 1]
            del extrinsic_diff
            torch.cuda.empty_cache()
        else:
            save_result_images(in_img, f'{args.eval_result_save_root}/scene.jpg')
            save_result_images(ex_img, f'{args.eval_result_save_root}/light.jpg')

        decoder = Decoder(in_channels=3, out_channels=3, latent_channels=174).to(device)
        decoder_optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
        decoder, _, _, _ = load_model(decoder, decoder_optimizer, save_path=args.decoder_path, device=device)

        decoder.eval()
        output_image = decoder(intrinsic_latent, extrinsic_latent)  # [1, 3, 256, 256]

        save_result_images(output_image, f'{args.eval_result_save_root}/result.jpg')

def eval(args, device="cuda"):
    print('\nstart evalution...')
    torch.cuda.empty_cache()

    with torch.inference_mode():
        # intrinsic diffusion
        intrinsic_diff = Diffusion(sample_size=128, in_channels=110).to(device)
        intrinsic_optimizer = optim.AdamW(intrinsic_diff.parameters(), lr=args.lr)
        intrinsic_diff, _, _, _ = load_model(intrinsic_diff, intrinsic_optimizer, save_path=args.intrinsic_path, device=device)

        intrinsic_diff.eval()
        intrinsic_latent = intrinsic_diff(torch.randn(1, 110, 128, 128).to(device), timestep=1000).sample  # [1, 110, 128, 128]
        del intrinsic_diff
        torch.cuda.empty_cache()

        # extrinsic diffusion
        extrinsic_diff = Diffusion(sample_size=128, in_channels=64, is_intrinsic=False).to(device)
        extrinsic_optimizer = optim.AdamW(extrinsic_diff.parameters(), lr=args.lr)
        extrinsic_diff, _, _, _ = load_model(extrinsic_diff, extrinsic_optimizer, save_path=args.extrinsic_path, device=device)
        
        extrinsic_diff.eval()
        extrinsic_latent = extrinsic_diff(torch.randn(1, 16, 1, 1).to(device), timestep=1000)  # [1, 16, 1, 1]
        del extrinsic_diff
        torch.cuda.empty_cache()

        # decoder
        decoder = Decoder(in_channels=3, out_channels=3, latent_channels=174).to(device)
        decoder_optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
        decoder, _, _, _ = load_model(decoder, decoder_optimizer, save_path=args.decoder_path, device=device)

        decoder.eval()
        output_image = decoder(intrinsic_latent, extrinsic_latent)  # [1, 3, 256, 256]

        save_result_images(output_image, f'{args.eval_result_save_root}/result.jpg')

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

    flatten_intrinsic = [tensor.flatten(start_dim=1) for tensor in intrinsic]
    cat_intrinsic = torch.cat(flatten_intrinsic, dim=1).view(intrinsic[0].shape[0], -1, 128, 128)    # shape: [batch, 1, 110, 128, 128]
    return cat_intrinsic, extrinsic

def load_latent_intrinsic(path, device):
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

def load_latent_intrinsic_multi(path):
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

    local_rank = int(os.environ["LOCAL_RANK"])  # 取得 local rank
    torch.cuda.set_device(local_rank)  # 設定每個進程對應的 GPU

    model = UNet(img_resolution = 256, in_channels = 3, out_channels = 3,
                     num_blocks_list = [1, 2, 2, 4, 4, 4], attn_resolutions = [0], model_channels = 32,
                     channel_mult = [1, 2, 4, 4, 8, 16], affine_scale = float(5e-3))
    model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)

    # load pretrained latent intrinsic model
    print(f"=> loading checkpoint '{path}' on GPU {local_rank}...")
    checkpoint = torch.load(path, map_location=f'cuda:{local_rank}')
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

def save_visualize_images(rec_img, tar_img):
    def tensor_to_image(tensor):
        """Convert a tensor to a NumPy image array."""
        img = (tensor.clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).cpu().data.numpy() * 255
        return img.astype(np.uint8)

    # Convert tensors to images
    rec_np = tensor_to_image(rec_img)
    tar_np = tensor_to_image(tar_img)

    # Concatenate along batch dimension (vertical stacking)
    for i, (rec, tar) in enumerate(zip(rec_np, tar_np)):
        concatenated_image = np.concatenate((rec, tar), axis=1)
        img = Image.fromarray(concatenated_image)

        # Save the concatenated image
        path = f"./visualize/epoch_{i}.png"
        print(f'=> saving image to {path}...')
        img.save(path)

def save_result_images(img_tensor, path):
    def tensor_to_image(tensor):
        """Convert a tensor to a NumPy image array."""
        tensor = tensor.squeeze(0)
        tensor = (tensor.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
        tensor = tensor.permute(1, 2, 0).cpu().numpy() * 255  # Convert to HWC format
        return tensor.astype(np.uint8)

    img_np = tensor_to_image(img_tensor)

    img = Image.fromarray(img_np)
    print(f"=> Saving image to {path}...")
    img.save(path)