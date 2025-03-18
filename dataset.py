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
        self.for_decoder=for_decoder
        self.device = f'cuda:{args.gpu_id}'
        self.model = model
        self.img_list = glob.glob(args.data_path + '/*.jpg')
        self.img_list.sort()
        print(f'=> initializng MIIWDataset... total count: {len(self.img_list)}')
        self.img_transform = transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
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