import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm
from PIL import Image
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

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

def train_img_diffusion(train_loader, args, device="cuda", save_root="./ckpt/intrinsic", resume=False):
    log_subdir = f'{logdir}/img_loss'
    os.makedirs(log_subdir, exist_ok=True)
    writer = SummaryWriter(log_subdir)

    model = Diffusion(sample_size=int(args.resize_size/2), in_channels=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    criterion = nn.MSELoss()
    start_epoch = 0
    
    if resume:
        ckpt_list = glob.glob(f'{save_root}/*.pth')
        ckpt_list.sort()
        model, optimizer, start_epoch, _ = load_model(model, optimizer, ckpt_list[-1], device)
    
    print('\nstart training image diffusion...')
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for image in pbar:  
            inputs = image.to(device) # shape: [batch, 3, resize_size, resize_size]
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

            model.eval()
            with torch.no_grad():
                # 從純噪聲開始，這裡需要和模型訓練時的尺寸一致
                noise = torch.randn(1, 3, int(args.resize_size/2), int(args.resize_size/2)).to(device)
                img = noise
                # 反向迭代每個 timestep (假設 scheduler 提供 timesteps 列表)
                for t in reversed(range(noise_scheduler.num_train_timesteps)):
                    # 可以使用 scheduler 提供的特定方法（例如：step）來更新圖片
                    noise_pred = model(img, torch.tensor([t], device=device)).sample
                    step_output = noise_scheduler.step(noise_pred, t, img)
                    img = step_output.prev_sample

                # 將生成的 img 處理並保存
                array = img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
                array = (array - array.min()) / (array.max() - array.min()) * 255
                array = array.astype(np.uint8)
                img_save_path = f'eval_{epoch+1}.jpg'
                Image.fromarray(array).save(img_save_path)
                print(f"Image saved at {img_save_path}")

def train_intrinsic_diffusion(train_loader, args, device="cuda", save_root="./ckpt/intrinsic", resume=False):
    log_subdir = f'{logdir}/intrinsic_loss'
    os.makedirs(log_subdir, exist_ok=True)
    writer = SummaryWriter(log_subdir)

    # Diffusion model
    model = Diffusion(sample_size=int(args.resize_size/2), in_channels=110).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = nn.MSELoss()

    noise_scheduler = DDIMScheduler(
        beta_start=1e-4, beta_end=0.02, beta_schedule="linear",
        num_train_timesteps=args.num_train_timesteps,
        prediction_type="epsilon"
    )
    noise_scheduler.set_timesteps(int(args.num_train_timesteps*0.011))

    start_epoch = 0
    if resume:
        ckpt_list = sorted(glob.glob(f'{save_root}/*.pth'))
        model, optimizer, start_epoch, _ = load_model(model, optimizer, ckpt_list[-1], device)

    # Decoder (frozen)
    decoder = Decoder2(in_channels=3, out_channels=3, latent_channels=174).to(device)
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)  # not used, but loader needs it
    ckpt_list = sorted(glob.glob(f'./ckpt/decoder/*.pth'))
    decoder, _, _, _ = load_model(decoder, decoder_optimizer, ckpt_list[-1], device)

    print('\nstart training intrinsic diffusion...')
    for epoch in range(start_epoch, args.num_epochs):
        model.train()  # ensure diffusion model is in training mode
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for intrinsic, extrinsic, img in pbar:
            # prepare inputs
            inputs = intrinsic.squeeze(1).to(device)     # [B,110,128,128]
            extrinsic = extrinsic.view(intrinsic.shape[0], -1, 1, 1).to(device)
            img = img.squeeze(1).to(device)

            optimizer.zero_grad()

            # forward diffusion + noise prediction loss
            noise = torch.randn_like(inputs)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (inputs.shape[0],), device=device)
            noisy_inputs = noise_scheduler.add_noise(inputs, noise, timesteps)
            pred_noise  = model(noisy_inputs, timestep=timesteps).sample
            diff_loss  = criterion(pred_noise, noise)

            # DDIM 多步逆扩散生成 latent
            with torch.no_grad():
                latent = noisy_inputs
                for t in noise_scheduler.timesteps:
                    int_t = int(t)
                    t_batch = torch.full((inputs.shape[0],), int_t, device=device, dtype=torch.long)
                    eps_pred = model(latent, timestep=t_batch).sample
                    latent = noise_scheduler.step(eps_pred, int_t, latent).prev_sample
                intrinsic_latent = latent

            # decode & reconstruction loss (decoder frozen, but grad flows to model)
            reconstructed_img = decoder(intrinsic_latent, extrinsic)
            recon_loss = criterion(reconstructed_img, img)

            # combined loss & backward
            loss = diff_loss + recon_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg:.6f}")
        writer.add_scalar('Intrinsic Loss', avg, epoch+1)

        if (epoch+1) % args.save_per_epoch == 0:
            save_model(model, optimizer, epoch+1, avg, f'{save_root}/checkpoint_{epoch+1:03d}.pth')

def train_extrinsic_diffusion(train_loader, args, device="cuda", save_root="./ckpt/extrinsic", resume=False):
    log_subdir = f'{logdir}/extrinsic_loss'
    os.makedirs(log_subdir, exist_ok=True)
    writer = SummaryWriter(log_subdir)

    model = Diffusion(sample_size=int(args.resize_size/2), in_channels=64, is_intrinsic=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
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

    decoder = Decoder2(in_channels=3, out_channels=3, latent_channels=174).to(device)
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

def test_distribution(args, dataloader, device="cuda"):
    log_subdir = f'{logdir}/new_distribution_step{args.num_train_timesteps}'
    os.makedirs(log_subdir, exist_ok=True)
    writer = SummaryWriter(log_subdir)

    print('\nstart test distribution...')
    torch.cuda.empty_cache()

    with torch.inference_mode():
        print('loading intrinsic diff...')
        intrinsic_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
        intrinsic_diff = Diffusion(sample_size=int(args.resize_size/2), in_channels=110).to(device)
        intrinsic_optimizer = optim.AdamW(intrinsic_diff.parameters(), lr=args.lr)
        intrinsic_diff, _, _, _ = load_model(intrinsic_diff, intrinsic_optimizer, save_path=args.intrinsic_path, device=device)

        intrinsic_diff.eval()
        pbar = tqdm(dataloader, desc=f"Processing")
        for i, (gt_latent, _, _) in enumerate(pbar):
            gt_latent = gt_latent.squeeze(1)

            sampled_latent_list = []
            for batch in range(args.batch_size):
                intrinsic_noise = torch.randn(1, 110, int(args.resize_size/2), int(args.resize_size/2)).to(device)
                for t in reversed(range(args.num_train_timesteps)):
                    t_tensor = torch.tensor([t], device=device)
                    pred_noise = intrinsic_diff(intrinsic_noise, timestep=t_tensor).sample
                    step_result = intrinsic_scheduler.step(pred_noise, t, intrinsic_noise)
                    intrinsic_noise = step_result.prev_sample
                sampled_latent_list.append(intrinsic_noise)
                
            sampled_latent = torch.cat(sampled_latent_list, 0)
            log_latent_analysis(writer, gt_latent, sampled_latent, i)
        del intrinsic_diff
        torch.cuda.empty_cache()            


def part_eval(args, pretrained_model, device="cuda"):
    print('\nstart part evalution...')
    torch.cuda.empty_cache()

    with torch.inference_mode():
        img_transform = preprocess(args.resize_size)
        in_img = Image.open('./datasets/my_data/ori_3.jpg').convert("RGB")
        ex_img = Image.open('./datasets/my_data/ref_3.jpg').convert("RGB")
        
        in_img = img_transform(in_img).unsqueeze(0).to(device)
        ex_img = img_transform(ex_img).unsqueeze(0).to(device)

        intrinsic_latent, _ = get_image_intrinsic_extrinsic(pretrained_model, in_img, args.resize_size)

        _, extrinsic_latent = get_image_intrinsic_extrinsic(pretrained_model, ex_img, args.resize_size)
        extrinsic_latent = extrinsic_latent.view(extrinsic_latent.shape[0], -1, 1, 1)

        if args.eval_mode == 'intrinsic':
            print('evaluate intrinsic diff...')
            save_result_images(ex_img, f'{args.eval_result_save_root}/light.jpg')
            intrinsic_diff = Diffusion(sample_size=int(args.resize_size/2), in_channels=110).to(device)
            intrinsic_optimizer = optim.AdamW(intrinsic_diff.parameters(), lr=args.lr)
            intrinsic_diff, _, _, _ = load_model(intrinsic_diff, intrinsic_optimizer, save_path=args.intrinsic_path, device=device)

            intrinsic_scheduler = DDIMScheduler(
                beta_start=1e-4, beta_end=0.02, beta_schedule="linear",
                num_train_timesteps=args.num_train_timesteps,
                prediction_type="epsilon"
            )
            intrinsic_scheduler.set_timesteps(num_inference_steps=int(args.num_train_timesteps*0.051))

            intrinsic_diff.eval()
            latent = torch.randn(1, 110, int(args.resize_size/2), int(args.resize_size/2)).to(device)
            for t in intrinsic_scheduler.timesteps:
                t_batch = torch.tensor([t], device=device)
                eps_pred = intrinsic_diff(latent, timestep=t_batch).sample
                latent = intrinsic_scheduler.step(eps_pred, t, latent).prev_sample
            intrinsic_latent = latent
            del intrinsic_diff
            torch.cuda.empty_cache()
        elif args.eval_mode == 'extrinsic':
            print('evaluate extrinsic diff...')
            save_result_images(in_img, f'{args.eval_result_save_root}/scene.jpg')
            extrinsic_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
            extrinsic_diff = Diffusion(sample_size=int(args.resize_size/2), in_channels=64, is_intrinsic=False).to(device)
            extrinsic_optimizer = optim.AdamW(extrinsic_diff.parameters(), lr=args.lr)
            extrinsic_diff, _, _, _ = load_model(extrinsic_diff, extrinsic_optimizer, save_path=args.extrinsic_path, device=device)
            
            extrinsic_diff.eval()
            extrinsic_noise = torch.randn(1, 16, 1, 1).to(device)
            for t in reversed(range(args.num_train_timesteps)):
                t_tensor = torch.tensor([t], device=device)
                pred_noise = extrinsic_diff(extrinsic_noise, timestep=t_tensor)
                step_result = extrinsic_scheduler.step(pred_noise, t, extrinsic_noise)
                extrinsic_noise = step_result.prev_sample
            extrinsic_latent = extrinsic_noise
            del extrinsic_diff
            torch.cuda.empty_cache()
        else:
            save_result_images(in_img, f'{args.eval_result_save_root}/scene.jpg')
            save_result_images(ex_img, f'{args.eval_result_save_root}/light.jpg')

        decoder = Decoder2(in_channels=3, out_channels=3, latent_channels=174).to(device)
        decoder_optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
        decoder, _, _, _ = load_model(decoder, decoder_optimizer, save_path=args.decoder_path, device=device)

        decoder.eval()
        output_image = decoder(intrinsic_latent, extrinsic_latent)  # [1, 3, 256, 256]
        save_result_images(output_image, f'{args.eval_result_save_root}/result.jpg')


def eval(args, device="cuda"):
    print('\nstart evaluation...')
    torch.cuda.empty_cache()

    # 建立 scheduler (這裡假設兩個 diffusion 模型都使用相同的 num_train_timesteps)
    intrinsic_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    extrinsic_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    with torch.inference_mode():
        # intrinsic diffusion
        intrinsic_diff = Diffusion(sample_size=int(args.resize_size/2), in_channels=110).to(device)
        intrinsic_optimizer = optim.AdamW(intrinsic_diff.parameters(), lr=args.lr)
        intrinsic_diff, _, _, _ = load_model(intrinsic_diff, intrinsic_optimizer, save_path=args.intrinsic_path, device=device)
        intrinsic_diff.eval()

        # 依照多步驟反向生成 intrinsic latent
        intrinsic_noise = torch.randn(1, 110, int(args.resize_size/2), int(args.resize_size/2)).to(device)
        for t in reversed(range(args.num_train_timesteps)):
            t_tensor = torch.tensor([t], device=device)
            # 模型預測噪聲
            pred_noise = intrinsic_diff(intrinsic_noise, timestep=t_tensor).sample
            # scheduler 根據預測噪聲更新圖像
            step_result = intrinsic_scheduler.step(pred_noise, t, intrinsic_noise)
            intrinsic_noise = step_result.prev_sample
        intrinsic_latent = intrinsic_noise

        del intrinsic_diff
        torch.cuda.empty_cache()

        # extrinsic diffusion
        extrinsic_diff = Diffusion(sample_size=int(args.resize_size/2), in_channels=64, is_intrinsic=False).to(device)
        extrinsic_optimizer = optim.AdamW(extrinsic_diff.parameters(), lr=args.lr)
        extrinsic_diff, _, _, _ = load_model(extrinsic_diff, extrinsic_optimizer, save_path=args.extrinsic_path, device=device)
        extrinsic_diff.eval()

        # 多步驟反向生成 extrinsic latent
        extrinsic_noise = torch.randn(1, 16, 1, 1).to(device)
        for t in reversed(range(args.num_train_timesteps)):
            t_tensor = torch.tensor([t], device=device)
            pred_noise = extrinsic_diff(extrinsic_noise, timestep=t_tensor)
            step_result = extrinsic_scheduler.step(pred_noise, t, extrinsic_noise)
            extrinsic_noise = step_result.prev_sample
        extrinsic_latent = extrinsic_noise

        del extrinsic_diff
        torch.cuda.empty_cache()

        # decoder
        decoder = Decoder(in_channels=3, out_channels=3, latent_channels=174).to(device)
        decoder_optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
        decoder, _, _, _ = load_model(decoder, decoder_optimizer, save_path=args.decoder_path, device=device)
        decoder.eval()
        output_image = decoder(intrinsic_latent, extrinsic_latent)  # 輸出圖像 shape: [1, 3, 256, 256]

        save_result_images(output_image, f'{args.eval_result_save_root}/result.jpg')

#----------------------------------------------------------------------------
# Utils
#----------------------------------------------------------------------------

def preprocess(resize_size):
    img_transform = transforms.Compose([
        torchvision.transforms.Resize(resize_size),
        torchvision.transforms.CenterCrop((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return img_transform

def get_image_intrinsic_extrinsic(model, img, resize_size):
    model.eval()

    P_mean=-0.5
    P_std=1.2
    rnd_normal = torch.randn([img.shape[0], 1, 1, 1], device=img.device)
    sigma = (rnd_normal * P_std + P_mean).exp().to(img.device) * 0 + 0.001

    noise = torch.randn_like(img)

    intrinsic, extrinsic = model(img, run_encoder = True)

    flatten_intrinsic = [tensor.flatten(start_dim=1) for tensor in intrinsic]
    cat_intrinsic = torch.cat(flatten_intrinsic, dim=1).view(intrinsic[0].shape[0], -1, int(resize_size/2), int(resize_size/2))    # shape: [batch, 1, 110, 128, 128]
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

def save_latent(dataloader, root):
    latent_dict_list = []
    pbar = tqdm(dataloader, desc='Processing')
    for i, (in_code, ex_code, path) in enumerate(pbar):
        latent_dict_list.append({"path": path, "intrinsic": in_code.squeeze(0).squeeze(0).cpu(), "extrinsic": ex_code.squeeze(0).squeeze(0).cpu()})

        if (i+1) % 40 == 0:
            torch.save(latent_dict_list, f"{root}/sub_{i+1:03d}.pt")
            latent_dict_list = []    

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

def log_latent_analysis(writer: SummaryWriter,
                        gt_latent: torch.Tensor,
                        sampled_latent: torch.Tensor,
                        step: int,
                        log_embedding: bool = True,
                        pca_dim: int = 50):
    """
    Logs distributional info of ground truth and sampled latent to TensorBoard.

    Parameters:
        writer: TensorBoard SummaryWriter
        gt_latent: Tensor of shape [B, C, H, W]
        sampled_latent: Tensor of shape [B, C, H, W]
        step: current global step
        log_embedding: whether to log 2D/3D embedding visualization
        pca_dim: dimension to reduce to before feeding to add_embedding
    """
    B = min(gt_latent.shape[0], sampled_latent.shape[0])
    gt_flat = gt_latent[:B].reshape(B, -1)
    sample_flat = sampled_latent[:B].reshape(B, -1)

    if gt_flat.shape[1] != sample_flat.shape[1]:
        print(f"[Error] Latent dims not aligned: gt={gt_flat.shape}, sample={sample_flat.shape}")
        return

    all_latents = torch.cat([gt_flat, sample_flat], dim=0)
    labels = ['gt'] * B + ['sample'] * B

    # Histogram + scalar
    writer.add_histogram("latent_gt", gt_latent, step)
    writer.add_histogram("latent_sample", sampled_latent, step)
    writer.add_scalar("latent_gt/mean", gt_latent.mean().item(), step)
    writer.add_scalar("latent_gt/std", gt_latent.std().item(), step)
    writer.add_scalar("latent_sample/mean", sampled_latent.mean().item(), step)
    writer.add_scalar("latent_sample/std", sampled_latent.std().item(), step)

    # Embedding
    if log_embedding:
        try:
            latent_np = all_latents.detach().cpu().numpy()
            max_dim = min(pca_dim, latent_np.shape[0], latent_np.shape[1])
            pca = PCA(n_components=max_dim)
            reduced_latents = torch.tensor(pca.fit_transform(latent_np), dtype=torch.float32)
            assert len(labels) == reduced_latents.shape[0], f"Metadata count {len(labels)} != latent count {reduced_latents.shape[0]}"
            writer.add_embedding(reduced_latents, metadata=labels, tag='latent_comparison', global_step=step)
        except Exception as e:
            print(f"[Warning] PCA failed: {e}")