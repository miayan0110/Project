import os
import glob
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from unets import UNet
import torch.distributed as dist
import numpy as np

#----------------------------------------------------------------------------
# Dataset
#----------------------------------------------------------------------------

class MIIWDataset(Dataset):
    def __init__(self, model, args, for_decoder=False):
        self.for_decoder=for_decoder
        self.device = f'cuda:{args.gpu_id}'
        self.model = model
        self.img_list = glob.glob(args.data_path + '/*.jpg')
        self.img_list.sort()
        print(f'=> initializng MIIWDataset... total count: {len(self.img_list)}')
        # self.img_transform = transforms.ToTensor()
        self.img_transform = transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])

    def __len__(self):
        # return len(self.img_list) // 2
        return len(self.img_list)

    def __getitem__(self, idx):
        # idx *= 2
        # lighting_path = self.img_list[idx]
        # content_path = self.img_list[idx+1]
        # print(f'lighting image path: {lighting_path}, content image path: {content_path}')

        # content_image = Image.open(content_path).convert("RGB")
        # lighting_image = Image.open(lighting_path).convert("RGB")

        # content_tensor = self.img_transform(content_image)
        # lighting_tensor = self.img_transform(lighting_image)

        # return content_tensor, lighting_tensor

        img_path = self.img_list[idx]
        # print(f'image path: {img_path}')
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img).unsqueeze(0).to(self.device)

        intrinsic, extrinsic = get_image_intrinsic_extrinsic(self.model, img)
        flatten_intrinsic = [tensor.flatten(start_dim=1) for tensor in intrinsic]
        cat_intrinsic = torch.cat(flatten_intrinsic, dim=1).view(intrinsic[0].shape[0], -1, 128, 128)    # shape: [batch, 1, 110, 128, 128]
        if self.for_decoder:
            return img, cat_intrinsic, extrinsic
        else:
            return cat_intrinsic, extrinsic

class StyLitGAN_Dataset(Dataset):
    # 先確認styltigan提供的data有多少
    def __init__(self, root):
        self.img_list = glob.glob(f'{root}/*.jpg')
        self.img_list.sort()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert("RGB")
        img = self.to_tensor(img)

        return img
    

#----------------------------------------------------------------------------
# Get Latents
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


#----------------------------------------------------------------------------
# Utils
#----------------------------------------------------------------------------

def load_latent_intrinsic(path, device=0):
    # initialize
    dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=1, rank=0)
    model = UNet(img_resolution = 256, in_channels = 3, out_channels = 3,
                     num_blocks_list = [1, 2, 2, 4, 4, 4], attn_resolutions = [0], model_channels = 32,
                     channel_mult = [1, 2, 4, 4, 8, 16], affine_scale = float(5e-3))
    model.cuda(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True,
    broadcast_buffers=False)

    # load pretrained latent intrinsic model
    print(f"=> loading checkpoint '{path}'...")
    checkpoint = torch.load(path, map_location=f'cuda:{device}')
    model.load_state_dict(checkpoint['state_dict'])
    print('=> finished.')

    return model

def load_model(model, optimizer, save_path="diffusion_checkpoint.pth", device="cuda"):
    # load checkpoints
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

def save_images(rec_img, tar_img):
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
        path = f"./visualize/batch_{i}.png"
        print(f'=> saving image to {path}...')
        img.save(path)


    
# if __name__ == '__main__':
#     dataset = MIIWDataset(args)
#     print(dataset[5])

    # dataset = StyleLitGAN_Dataset('./datasets/style_lit_gan')
    # print(dataset[0])


    # model = load_latent_intrinsic('./pretrained_weight/latent_intrinsic.pth.tar', 7)