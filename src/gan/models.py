import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import torchvision.models as models


# Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=4):
        super(MappingNetwork, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2, inplace=True)]
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# Modulated Convolution
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, w_dim=512, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.style_fc = nn.Linear(w_dim, in_channels)
        fan_in = in_channels * kernel_size**2
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(fan_in)))
        nn.init.kaiming_normal_(self.conv.weight, a=0.2)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, w):
        B = x.size(0)
        s = self.style_fc(w).view(B, 1, self.in_channels, 1, 1)
        weight = self.conv.weight * self.scale
        weight = weight.unsqueeze(0) * s
        demod = torch.rsqrt((weight**2).sum([2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * demod
        weight = weight.view(
            B * self.out_channels, self.in_channels, *self.conv.kernel_size
        )
        x = x.view(1, B * self.in_channels, x.size(2), x.size(3))
        out = F.conv2d(x, weight, padding=self.conv.padding, groups=B)
        return out.view(B, self.out_channels, out.size(2), out.size(3))


# Synthesis Block
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, final_resolution=256):
        super().__init__()
        self.resolution = resolution
        self.conv1 = ModulatedConv2d(in_channels, out_channels)
        self.conv2 = ModulatedConv2d(out_channels, out_channels)
        self.to_rgb = ModulatedConv2d(out_channels, 3, kernel_size=1, padding=0)
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if resolution > 4
            else nn.Identity()
        )
        self.rgb_upsample = nn.Upsample(
            size=(final_resolution, final_resolution),
            mode="bilinear",
            align_corners=False,
        )
        self.noise_scale1 = nn.Parameter(torch.zeros(1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, w, noise1=None, noise2=None):
        B, _, H, W = x.shape
        # Generate noise1 with out_channels (same as conv1 output)
        noise1 = noise1 or torch.randn(
            B, self.conv1.out_channels, H, W, device=x.device
        )
        x = self.conv1(x, w) + self.noise_scale1 * noise1
        x = F.leaky_relu(x, 0.2)
        # Generate noise2 with out_channels (same as conv2 output)
        noise2 = noise2 or torch.randn(
            B, self.conv2.out_channels, H, W, device=x.device
        )
        x = self.conv2(x, w) + self.noise_scale2 * noise2
        x = F.leaky_relu(x, 0.2)
        x = self.upsample(x)
        rgb = self.rgb_upsample(self.to_rgb(x, w))
        return x, rgb


# Generator
class Generator(nn.Module):
    def __init__(
        self,
        z_dim=512,
        w_dim=512,
        num_channels=[1024, 1024, 640, 640, 320, 320, 160, 80],
        final_resolution=256,
    ):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=8)
        self.const = nn.Parameter(torch.randn(1, num_channels[0], 4, 4))
        self.blocks = nn.ModuleList(
            [
                SynthesisBlock(
                    num_channels[i],
                    num_channels[i + 1],
                    4 * 2 ** (i + 1),
                    final_resolution,
                )
                for i in range(len(num_channels) - 1)
            ]
        )
        self.register_buffer("w_avg", torch.zeros(w_dim))
        self.truncation_beta = 0.995

    def forward(self, z, truncation_psi=1.0):
        w = self.mapping(z)
        if self.training:
            self.w_avg.copy_(
                self.truncation_beta * self.w_avg
                + (1 - self.truncation_beta) * w.mean(0)
            )
        elif truncation_psi < 1.0:
            w = truncation_psi * w + (1 - truncation_psi) * self.w_avg

        x = self.const.repeat(z.size(0), 1, 1, 1)
        rgb = None
        for block in self.blocks:
            x, rgb_i = block(x, w)
            rgb = rgb_i if rgb is None else rgb + rgb_i
        return torch.tanh(rgb)


class MinibatchStdDev(nn.Module):
    def forward(self, x):
        batch, c, h, w = x.shape
        std = x.std(dim=0, keepdim=True).mean().view(1, 1, 1, 1).expand(batch, 1, h, w)
        return torch.cat([x, std], dim=1)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=[64, 128, 256, 512, 768, 896]):
        super().__init__()
        self.blur = nn.AvgPool2d(2, stride=1)
        layers = []
        ch = in_channels
        for out_ch in base_channels:
            layers.extend(
                [
                    nn.Conv2d(ch, out_ch, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            ch = out_ch
        layers.append(MinibatchStdDev())
        layers.append(nn.Conv2d(ch + 1, ch, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(ch * 4 * 4, 1)

    def forward(self, x):
        x = self.blur(x)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Adaptive Discriminator Augmentation (ADA)
class ADA:
    def __init__(self, aug_prob=0.0):
        self.aug_prob = aug_prob
        self.aug = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )

    def update(self, real_scores):
        p = torch.sign(real_scores).mean().item()
        if p > 0.6:
            self.aug_prob = min(1.0, self.aug_prob + 0.01)
        elif p < 0.6:
            self.aug_prob = max(0.0, self.aug_prob - 0.01)

    def apply(self, x):
        mask = torch.rand(x.size(0), device=x.device) < self.aug_prob
        if mask.any():
            x_aug = [self.aug(img) if m else img for img, m in zip(x, mask)]
            x = torch.stack(x_aug)
        return x


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, fake, real):
        fake_vgg = self.vgg(fake)
        real_vgg = self.vgg(real)
        return F.l1_loss(fake_vgg, real_vgg)
