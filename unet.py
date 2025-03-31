import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == '__main__':
    import torch
    x = torch.randn((3,1,160,160))
    model = UNet()
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert x.shape == preds.shape