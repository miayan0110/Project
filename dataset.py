import glob
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from model_utils import *


img_transform = transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

#----------------------------------------------------------------------------
# Dataset
#----------------------------------------------------------------------------

class MIIWDataset(Dataset):
    def __init__(self, model, args, for_decoder=False):
        self.for_decoder = for_decoder
        self.device = f'cuda:{args.gpu_id}'
        self.model = model
        self.img_transform = img_transform
        self.img_list = glob.glob(args.data_path + '/*.jpg')
        self.img_list.sort()
        print(f'=> initializng MIIWDataset... total count: {len(self.img_list)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img).unsqueeze(0).to(self.device)

        cat_intrinsic, extrinsic = get_image_intrinsic_extrinsic(self.model, img)

        if self.for_decoder:
            return img, cat_intrinsic, extrinsic
        else:
            return cat_intrinsic, extrinsic

class StyLitGAN_Dataset(Dataset):
    # 先確認styltigan提供的data有多少：7000筆
    def __init__(self, model, args, for_decoder=False):
        self.for_decoder = for_decoder
        self.device = f'cuda:{args.gpu_id}'
        self.data_root = args.data_path
        self.model = model
        self.img_transform = img_transform

        self.img_list = glob.glob(args.data_path + '/*.jpg')
        print(f'=> initializng StyLitGAN Dataset... total count: {int(len(self.img_list) * 7 / 8)}')

    def __len__(self):
        return int(len(self.img_list) * 7 / 8)

    def __getitem__(self, idx):
        scene_id = int(idx / 7)
        light_id = idx % 7 + 1
        scene_img_path = f'{self.data_root}/{scene_id}_0.jpg'
        light_img_path = f'{self.data_root}/{scene_id}_{light_id}.jpg'

        scene_img = Image.open(scene_img_path).convert("RGB")
        light_img = Image.open(light_img_path).convert("RGB")

        scene_img = self.img_transform(scene_img).unsqueeze(0).to(self.device)
        light_img = self.img_transform(light_img).unsqueeze(0).to(self.device)
        
        scene_cat_intrinsic, scene_extrinsic = get_image_intrinsic_extrinsic(self.model, scene_img)
        light_cat_intrinsic, light_extrinsic = get_image_intrinsic_extrinsic(self.model, light_img)
        if self.for_decoder:
            return scene_img, light_img, scene_cat_intrinsic, scene_extrinsic, light_extrinsic
        else:
            return scene_cat_intrinsic, light_extrinsic