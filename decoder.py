import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import torch.nn.functional as F  # en haut du fichier si pas déjà fait

from double_conv import DoubleConv2D, DoubleConv3D


class Decoder2D(nn.Module):

    def __init__(self, out_channel=1, channels=(512, 256, 128, 64)):
        super(Decoder2D, self).__init__()

        self.up = nn.ModuleList()

        for channel in channels:
            # upsampling convolution
            self.up.append(
                nn.ConvTranspose2d(in_channels=channel * 2, out_channels=channel, kernel_size=2, stride=2)
            )
            self.up.append(DoubleConv2D(channel * 2, channel))

        self.final_conv = nn.Conv2d(channels[-1], out_channel, kernel_size=1)

    def forward(self, x):
        out, skip_connections = x

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.up), 2):
            out = self.up[i](out)
            skip_connection = skip_connections[i//2] # can't do i-1 because of index 0

            # avoids problems arising from the difference of size
            if out.shape != skip_connection:
                out = tf.resize(out, size=skip_connection.shape[2:])

            concat = torch.cat((skip_connection, out), dim=1)

            out = self.up[i+1](concat)

        return self.final_conv(out)
    


class Decoder3D(nn.Module):

    def __init__(self, out_channel=1, channels=(512, 256, 128, 64)):
        super(Decoder3D, self).__init__()

        self.up = nn.ModuleList()

        for channel in channels:
            # upsampling convolution
            self.up.append(
                nn.ConvTranspose3d(in_channels=channel * 2, out_channels=channel, kernel_size=2, stride=2)
            )
            self.up.append(DoubleConv3D(channel * 2, channel))

        self.final_conv = nn.Conv3d(channels[-1], out_channel, kernel_size=1)

    def forward(self, x):
        out, skip_connections = x

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.up), 2):
            out = self.up[i](out)
            skip_connection = skip_connections[i//2] # can't do i-1 because of index 0

            # avoids problems arising from the difference of size
            if out.shape != skip_connection:
                out = F.interpolate(out, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)

            concat = torch.cat((skip_connection, out), dim=1)

            out = self.up[i+1](concat)

        return self.final_conv(out)