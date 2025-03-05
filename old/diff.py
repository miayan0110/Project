import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from utils import *

class Diffusion(nn.Module):
    def __init__(self, sample_size, in_channels, is_intrinsic=True, is_eval=False):
        super().__init__()
        self.is_intrinsic = is_intrinsic
        self.is_eval = is_eval
        self.model = UNet2DModel(
            sample_size=sample_size,  # Adjust based on input resolution
            in_channels=in_channels,   # Input tensor shape: [batch, 64, 128, 128]
            out_channels=in_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 1024, 2048),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
            )
        )
        if not self.is_intrinsic:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(16, 64, kernel_size=4, stride=4),  # [batch, 16, 1, 1] → [batch, 64, 4, 4]
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4),  # [batch, 64, 4, 4] → [batch, 64, 16, 16]
                nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8),  # [batch, 64, 16, 16] → [batch, 64, 128, 128]
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=4, stride=4),  # [batch, 64, 128, 128] → [batch, 64, 32, 32]
                nn.Conv2d(64, 64, kernel_size=4, stride=4),  # [batch, 64, 32, 32] → [batch, 64, 8, 8]
                nn.Conv2d(64, 16, kernel_size=8, stride=8)   # [batch, 64, 8, 8] → [batch, 16, 1, 1]
            )

    def forward(self, noisy_x, timestep):
        if not self.is_intrinsic:
            noisy_x = self.upsample(noisy_x)
            noisy_x = self.model(noisy_x, timestep).sample
            if not self.is_eval:
                noisy_x = self.downsample(noisy_x)
        else:
            noisy_x = self.model(noisy_x, timestep)
        return noisy_x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels):
        super().__init__()
        vae = AutoencoderKL(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=(128, 256),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D")
        )

        self.decoder = vae.decoder
        self.upsample = nn.Sequential(
                nn.ConvTranspose2d(16, 64, kernel_size=4, stride=4),  # [batch, 16, 1, 1] → [batch, 64, 4, 4]
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4),  # [batch, 64, 4, 4] → [batch, 64, 16, 16]
                nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8),  # [batch, 64, 16, 16] → [batch, 64, 128, 128]
            )

    def forward(self, intrinsic, extrinsic):
        extrinsic = extrinsic.view(extrinsic.shape[0], -1, 1, 1)    # [batch, 16, 1, 1]
        extrinsic = self.upsample(extrinsic)    # [batch, 64, 128, 128]
        feature = torch.cat([intrinsic, extrinsic], dim=1)  # [batch, 174, 128, 128]
        x = self.decoder(feature)     # [batch, 3, 256, 256]
        return x

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
            # for intr in intrinsic:
            #     print(intr.shape)
            # flatten_intrinsic = [tensor.to(device).flatten(start_dim=2) for tensor in intrinsic]
            # inputs = torch.cat(flatten_intrinsic, dim=2).view(intrinsic[0].shape[0], 1, -1, 128, 128).squeeze(1)    
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
            save_images(reconstructed_img, target_img)


def eval(device="cuda"):
    intrinsic_diff = Diffusion(sample_size=128, in_channels=110).to(device)
    extrinsic_diff = Diffusion(sample_size=128, in_channels=64, is_intrinsic=False).to(device)
    decoder = Decoder(in_channels=3, out_channels=3, latent_channels=174).to(device)
    load_model()