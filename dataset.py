import glob
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from model_utils import *

#----------------------------------------------------------------------------
# Dataset
#----------------------------------------------------------------------------

class MIIWDataset(Dataset):
    def __init__(self, model, args, for_decoder=False):
        self.for_decoder = for_decoder
        self.device = f'cuda:{args.gpu_id}'
        self.model = model
        self.img_transform = preprocess(args.resize_size)
        self.resize_size = args.resize_size
        self.img_list = glob.glob(args.data_path + '/*.jpg')
        self.img_list.sort()
        print(f'=> initializng MIIWDataset... total count: {len(self.img_list)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img).unsqueeze(0).to(self.device)

        cat_intrinsic, extrinsic = get_image_intrinsic_extrinsic(self.model, img, self.resize_size)

        if self.for_decoder:
            return img, cat_intrinsic, extrinsic
        else:
            return cat_intrinsic, extrinsic

class StyLitGAN_Dataset(Dataset):
    def __init__(self, model, args, for_decoder=False):
        self.for_decoder = for_decoder
        self.device = f'cuda:{args.gpu_id}'
        self.data_root = args.data_path
        self.model = model
        self.img_transform = preprocess(args.resize_size)
        self.resize_size = args.resize_size

        self.img_list = glob.glob(args.data_path + '/*.jpg')
        
        self.len = len(self.img_list)
        if self.for_decoder:
            self.len = int(len(self.img_list) * 7 / 8)

        print(f'=> initializng StyLitGAN Dataset... total count: {self.len}')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.for_decoder:
            scene_id = int(idx / 7)
            light_id = idx % 7 + 1
            scene_img_path = f'{self.data_root}/{scene_id}_0.jpg'
            light_img_path = f'{self.data_root}/{scene_id}_{light_id}.jpg'

            scene_img = Image.open(scene_img_path).convert("RGB")
            light_img = Image.open(light_img_path).convert("RGB")

            scene_img = self.img_transform(scene_img).unsqueeze(0).to(self.device)
            light_img = self.img_transform(light_img).unsqueeze(0).to(self.device)
            
            scene_cat_intrinsic, scene_extrinsic = get_image_intrinsic_extrinsic(self.model, scene_img, self.resize_size)
            light_cat_intrinsic, light_extrinsic = get_image_intrinsic_extrinsic(self.model, light_img, self.resize_size)
            return scene_img, light_img, scene_cat_intrinsic, scene_extrinsic, light_extrinsic
        else:
            img_path = self.img_list[idx]
            img = Image.open(img_path).convert("RGB")
            img = self.img_transform(img).unsqueeze(0).to(self.device)
            intrinsic, extrinsic = get_image_intrinsic_extrinsic(self.model, img, self.resize_size)
            return intrinsic, extrinsic


class LSUNDataset(Dataset):
    def __init__(self, model, args):
        super().__init__()
        self.device = f'cuda:{args.gpu_id}'
        self.model = model
        self.img_transform = preprocess(args.resize_size)
        self.resize_size = args.resize_size
        
        self.img_list = glob.glob(f'{args.data_path}/*.jpg')[0:10000]
        # self.img_list = glob.glob(f'{args.data_path}/*.jpg')
        print(f'=> initializng LSUNDataset... total count: {len(self.img_list)}')

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img).unsqueeze(0).to(self.device)

        cat_intrinsic, extrinsic = get_image_intrinsic_extrinsic(self.model, img, self.resize_size)

        return cat_intrinsic, extrinsic