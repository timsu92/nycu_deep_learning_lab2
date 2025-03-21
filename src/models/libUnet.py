# https://ithelp.ithome.com.tw/articles/10240314
# https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):
    """Upscaling, concatenation of encoder features and double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_lower, x_encoder):
        x_lower = self.up(x_lower)

        # CHW
        diffH = x_encoder.size(-2) - x_lower.size(-2)
        diffW = x_encoder.size(-1) - x_lower.size(-1)

        # original feature map is at the center
        x_lower = F.pad(
            x_lower, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2]
        )

        # concat and increase channel
        x = torch.cat([x_encoder, x_lower], dim=-3)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
