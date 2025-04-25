import torch.nn as nn
from encoder import Encoder2D, Encoder3D
from decoder import Decoder2D, Decoder3D

class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder2D()
        self.decoder = Decoder2D()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D()
        self.decoder = Decoder3D()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == '__main__':
    import torch

    # 2D
    """
    x = torch.randn((3,1,160,160))
    model = UNet2D()
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert x.shape == preds.shape
    """

    # 3D
    x = torch.randn((1, 1, 116, 132, 132))
    model = UNet3D()
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert x.shape == preds.shape