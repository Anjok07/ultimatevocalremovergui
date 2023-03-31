import torch
import torch.nn as nn


class TFC(nn.Module):
    def __init__(self, c, l, k, norm):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    norm(c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


class DenseTFC(nn.Module):
    def __init__(self, c, l, k, norm):
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    norm(c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)


class TFC_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, dense=False, bias=True, norm=nn.BatchNorm2d):

        super(TFC_TDF, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c, l, k, norm) if dense else TFC(c, l, k, norm)

        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    norm(c),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    norm(c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    norm(c),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x) if self.use_tdf else x

