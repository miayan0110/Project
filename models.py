import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL

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