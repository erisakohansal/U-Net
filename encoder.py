import torch.nn as nn
from double_conv import DoubleConv2D, DoubleConv3D


class Encoder2D(nn.Module):

    def __init__(self, in_channels=1, channels=(64, 128, 256, 512)):
        # in_channel=1 for grayscale, 3 for RGB
        super(Encoder2D, self).__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv2D(in_channels, channel))
            in_channels = channel

        self.bottleneck = DoubleConv2D(channels[-1], channels[-1]*2)

    def forward(self, x):
        skip_connections = []

        for down in self.down:
            x = down(x)
            # skip connections store the feature maps from each level before they get downsampled
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        return x, skip_connections
    

class Encoder3D(nn.Module):

    def __init__(self, in_channels=1, channels=(64, 128, 256, 512)):
        # in_channel=1 for grayscale, 3 for RGB
        super(Encoder3D, self).__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv3D(in_channels, channel))
            in_channels = channel

        self.bottleneck = DoubleConv3D(channels[-1], channels[-1]*2)

    def forward(self, x):
        skip_connections = []

        for down in self.down:
            x = down(x)
            # skip connections store the feature maps from each level before they get downsampled
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        return x, skip_connections