import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel


class LinearResBlock(nn.Module):
    def __init__(self, dim):
        super(LinearResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class SmallDenoiserNetwork(nn.Module):
    def __init__(self, hidden_dim=64, resblocks=2, global_skip_connection=False):
        super(SmallDenoiserNetwork, self).__init__()
        self.global_residual = global_skip_connection
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            *[LinearResBlock(hidden_dim) for _ in range(resblocks)],
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t, t):
        t = t.float().view(-1, 1) / 1000.0
        net_input = torch.cat([x_t, t], dim=1)
        if not self.global_residual:
            return self.net(net_input)
        return self.net(net_input) + x_t


class LargeConvDenoiserNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: list = [64, 128, 256, 512, 1024],
        layers_per_block: int = 2,
        downblock: str = "ResnetDownsampleBlock2D",
        upblock: str = "ResnetUpsampleBlock2D",
        add_attention: bool = True,
        attention_head_dim: int = 64,
        low_condition: bool = False,
        timestep_condition: bool = True,
        global_skip_connection: bool = True,
        num_class_embeds: int = None,
    ):
        super().__init__()
        self.low_condition = low_condition
        self.timestep_condition = timestep_condition
        self.global_skip_connection = global_skip_connection
        self.divide_factor = 2 ** len(channels)

        in_channels = 2 * in_channels if self.low_condition else in_channels

        self.backbone = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=channels,
            layers_per_block=layers_per_block,
            down_block_types=tuple(downblock for _ in range(len(channels))),
            up_block_types=tuple(upblock for _ in range(len(channels))),
            add_attention=add_attention,
            attention_head_dim=attention_head_dim,
            num_class_embeds=num_class_embeds,
        )

    def padding(self, x):
        _, _, W, H = x.shape
        desired_width = (
            (W + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor
        desired_height = (
            (H + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor

        padding_w = desired_width - W
        padding_h = desired_height - H

        return F.pad(x, (0, padding_h, 0, padding_w), mode="constant", value=0), W, H

    def remove_padding(self, x, W, H):
        return x[:, :, :W, :H]

    def forward(self, x_t, t):
        x_in, W, H = self.padding(x_t)

        model_output = self.backbone(
            x_in,
            timestep=t if self.timestep_condition else 0,
        ).sample

        model_output = self.remove_padding(model_output, W, H)

        if self.global_skip_connection:
            model_output[:, :3] = model_output[:, :3] + x_t

        return model_output
