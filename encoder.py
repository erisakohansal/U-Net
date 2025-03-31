import torch.nn as nn
from double_conv import DoubleConv


class Encoder(nn.Module):

    def __init__(self, in_channels=1, channels=(64, 128, 256, 512)):
        # in_channel=1 for grayscale, 3 for RGB
        super(Encoder, self).__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv(in_channels, channel))
            in_channels = channel

        self.bottleneck = DoubleConv(512, 1024)

    def forward(self, x):
        skip_connections = []

        for down in self.down:
            x = down(x)
            # skip connections store the feature maps from each level before they get downsampled
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        return x, skip_connections