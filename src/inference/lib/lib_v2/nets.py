import torch
from torch import nn

from . import layers


class BaseASPPNet(nn.Module):

    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):

    def __init__(self):
        super(CascadedASPPNet, self).__init__()
        self.low_band_net = BaseASPPNet(2, 32, ((2, 4), (4, 8), (8, 16)))
        self.high_band_net = BaseASPPNet(2, 32, ((2, 4), (4, 8), (8, 16)))

        self.bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.full_band_net = BaseASPPNet(16, 32)

        self.out = nn.Sequential(
            layers.Conv2DBNActiv(32, 16, 3, 1, 1),
            nn.Conv2d(16, 2, 1, bias=False))
        self.aux_out = nn.Conv2d(32, 2, 1, bias=False)

        self.offset = 128

    def __call__(self, x):
        bandw = x.size()[2] // 2
        aux = torch.cat([
            self.low_band_net(x[:, :, :bandw]),
            self.high_band_net(x[:, :, bandw:])
        ], dim=2)

        h = torch.cat([x, aux], dim=1)
        h = self.full_band_net(self.bridge(h))

        h = torch.sigmoid(self.out(h))
        aux = torch.sigmoid(self.aux_out(aux))

        return h, aux

    def predict(self, x):
        bandw = x.size()[2] // 2
        aux = torch.cat([
            self.low_band_net(x[:, :, :bandw]),
            self.high_band_net(x[:, :, bandw:])
        ], dim=2)

        h = torch.cat([x, aux], dim=1)
        h = self.full_band_net(self.bridge(h))

        h = torch.sigmoid(self.out(h))
        if self.offset > 0:
            h = h[:, :, :, self.offset:-self.offset]
            assert h.size()[3] > 0

        return h
