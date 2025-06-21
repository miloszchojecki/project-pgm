import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

# Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=256, w_dim=256, num_layers=2):
        super(MappingNetwork, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(z_dim, w_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        w = self.net(z)  # Shape: [batch_size, w_dim]
        return w

# Modulated Convolution
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, w_dim=256, padding=1):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.w_dim = w_dim
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.style_fc = nn.Linear(w_dim, in_channels)
        fan_in = in_channels * kernel_size ** 2
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(fan_in)))
        nn.init.kaiming_normal_(self.conv.weight, a=0.2)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, w):
        batch_size = x.size(0)
        s = self.style_fc(w).view(batch_size, 1, self.in_channels, 1, 1)
        weight = self.conv.weight * self.scale
        weight = weight.unsqueeze(0) * s
        demod = torch.rsqrt((weight ** 2).sum([2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * demod
        weight = weight.view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, batch_size * self.in_channels, x.size(2), x.size(3))
        out = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        out = out.view(batch_size, self.out_channels, out.size(2), out.size(3))
        return out
    

# Synthesis Block
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, final_resolution=256):
        super(SynthesisBlock, self).__init__()
        self.conv1 = ModulatedConv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, kernel_size=3)
        self.to_rgb = ModulatedConv2d(out_channels, 3, kernel_size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if resolution > 4 else nn.Identity()
        self.rgb_upsample = nn.Upsample(size=(final_resolution, final_resolution), mode='bilinear', align_corners=False)        
        self.noise_scale1 = nn.Parameter(torch.zeros(1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1))
        self.out_channels = out_channels  # Store for noise generation

    def forward(self, x, w, noise1=None, noise2=None):
        # Generate noise with output channels of conv1 and conv2
        if noise1 is None:
            noise1 = torch.randn(x.size(0), self.out_channels, x.size(2), x.size(3), device=x.device)
        if noise2 is None:
            noise2 = torch.randn(x.size(0), self.out_channels, x.size(2), x.size(3), device=x.device)

        x = self.conv1(x, w)
        x = x + self.noise_scale1 * noise1
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x, w)
        x = x + self.noise_scale2 * noise2
        x = F.leaky_relu(x, 0.2)

        x = self.upsample(x)
        rgb = self.to_rgb(x, w)
        rgb = self.rgb_upsample(rgb)
        return x, rgb

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=256, w_dim=256, num_channels=[256, 256, 256, 128, 64, 32, 16], final_resolution=256):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, num_channels[0], 4, 4))
        self.blocks = nn.ModuleList([
            SynthesisBlock(num_channels[i], num_channels[i+1], 4 * 2**(i+1), final_resolution)
            for i in range(len(num_channels)-1)
        ])
        self.register_buffer('w_avg', torch.zeros(w_dim))
        self.truncation_beta = 0.995

    def forward(self, z, truncation_psi=1.0):
        w = self.mapping(z)
        if self.training:
            self.w_avg.copy_(self.truncation_beta * self.w_avg + (1 - self.truncation_beta) * w.detach().mean(0))
        else:
            if truncation_psi < 1.0:
                w = truncation_psi * w + (1 - truncation_psi) * self.w_avg

        x = self.const.repeat(z.size(0), 1, 1, 1)
        rgb = None
        for block in self.blocks:
            x, rgb_i = block(x, w)
            rgb = rgb_i if rgb is None else rgb + rgb_i
        return torch.tanh(rgb)
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_channels=[16, 32, 64, 128, 256, 256]):
        super(Discriminator, self).__init__()
        layers = []
        in_channels = 3
        for out_channels in num_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1] * 4 * 4, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Adaptive Discriminator Augmentation (ADA)
class ADA:
    def __init__(self, aug_prob=0.0):
        self.aug_prob = aug_prob
        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

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
