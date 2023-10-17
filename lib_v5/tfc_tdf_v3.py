import torch
import torch.nn as nn
from functools import partial

class STFT:
    def __init__(self, n_fft, hop_length, dim_f, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = dim_f
        self.device = device

    def __call__(self, x):
        
        x_is_mps = not x.device.type in ["cuda", "cpu"]
        if x_is_mps:
            x = x.cpu()

        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True,return_complex=False)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])

        if x_is_mps:
            x = x.to(self.device)

        return x[..., :self.dim_f, :]

    def inverse(self, x):
        
        x_is_mps = not x.device.type in ["cuda", "cpu"]
        if x_is_mps:
            x = x.cpu()

        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])

        if x_is_mps:
            x = x.to(self.device)

        return x

def get_norm(norm_type):
    def norm(c, norm_type):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type == 'InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)


def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type[:3] == 'elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    else:
        raise Exception


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class TFC_TDF_net(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        norm = get_norm(norm_type=config.model.norm)
        act = get_act(act_type=config.model.act)

        self.num_target_instruments = 1 if config.training.target_instrument else len(config.training.instruments)
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        n = config.model.num_scales
        scale = config.model.scale
        l = config.model.num_blocks_per_scale
        c = config.model.num_channels
        g = config.model.growth
        bn = config.model.bottleneck_factor
        f = config.audio.dim_f // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act)
            block.downscale = Downscale(c, c + g, scale, norm, act)
            f = f // scale[1]
            c += g
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(c, c, l, f, bn, norm, act)

        self.decoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.upscale = Upscale(c, c - g, scale, norm, act)
            f = f * scale[1]
            c -= g
            block.tfc_tdf = TFC_TDF(2 * c, c, l, f, bn, norm, act)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = STFT(config.audio.n_fft, config.audio.hop_length, config.audio.dim_f, self.device)

    def cac2cws(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        return x

    def cws2cac(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x

    def forward(self, x):

        x = self.stft(x)

        mix = x = self.cac2cws(x)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)

        return x


