import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

is_using_other_gpu = lambda device: not (device == "cpu" or device.startswith("cuda"))

class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, T)
        B, N, T = input.shape
        assert N % self.groups == 0

        #input.to("mps")

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input.to(self.weight.device)).reshape(B, N, T) * self.weight.reshape(1, -1, 1)
    
class RMVN(nn.Module):
    """
    Rescaled MVN.
    """
    def __init__(self, dimension, groups=1):
        super(RMVN, self).__init__()
        
        self.mean = nn.Parameter(torch.zeros(dimension))
        self.std = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, *)
        B, N = input.shape[:2]
        assert N % self.groups == 0
        input_reshape = input.reshape(B, self.groups, N // self.groups, -1)
        T = input_reshape.shape[-1]

        input_norm = (input_reshape - input_reshape.mean(2).unsqueeze(2)) / (input_reshape.var(2).unsqueeze(2) + self.eps).sqrt()
        input_norm = input_norm.reshape(B, N, T) * self.std.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)

        return input_norm.reshape(input.shape)
    
class Roformer(nn.Module):
    """
    Transformer with rotary positional embedding.
    """
    def __init__(self, input_size, hidden_size, num_head=8, theta=10000, window=10000, 
                 input_drop=0., attention_drop=0., causal=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.theta = theta  # base frequency for RoPE
        self.window = window
        # pre-calculate rotary embeddings
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)  # win, N
        self.register_buffer("sin_freq", sin_freq)  # win, N
        
        self.attention_drop = attention_drop
        self.causal = causal
        self.eps = 1e-5

        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size*self.num_head*3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size*self.num_head, self.input_size, 1, bias=False)

        self.MLP = nn.Sequential(RMSNorm(self.input_size),
                                 nn.Conv1d(self.input_size, self.input_size*8, 1, bias=False),
                                 nn.SiLU()
                                )
        self.MLP_output = nn.Conv1d(self.input_size*4, self.input_size, 1, bias=False)

    def _calc_rotary_emb(self):
        freq = 1. / (self.theta ** (torch.arange(0, self.hidden_size, 2)[:(self.hidden_size // 2)] / self.hidden_size))  # theta_i
        freq = freq.reshape(1, -1)  # 1, N//2
        pos = torch.arange(0, self.window).reshape(-1, 1)  # win, 1
        cos_freq = torch.cos(pos*freq)  # win, N//2
        sin_freq = torch.sin(pos*freq)  # win, N//2
        cos_freq = torch.stack([cos_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N
        sin_freq = torch.stack([sin_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N

        return cos_freq, sin_freq
    
    def _add_rotary_emb(self, feature, pos):
        # feature shape: ..., N
        N = feature.shape[-1]

        feature_reshape = feature.reshape(-1, N)
        pos = min(pos, self.window-1)
        cos_freq = self.cos_freq[pos]
        sin_freq = self.sin_freq[pos]
        reverse_sign = torch.from_numpy(np.asarray([-1, 1])).to(feature.device).type(feature.dtype)
        feature_reshape_neg = (torch.flip(feature_reshape.reshape(-1, N//2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(-1, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
    
        return feature_rope.reshape(feature.shape)

    def _add_rotary_sequence(self, feature):
        # feature shape: ..., T, N
        T, N = feature.shape[-2:]
        feature_reshape = feature.reshape(-1, T, N)

        cos_freq = self.cos_freq[:T]
        sin_freq = self.sin_freq[:T]
        reverse_sign = torch.from_numpy(np.asarray([-1, 1])).to(feature.device).type(feature.dtype)
        feature_reshape_neg = (torch.flip(feature_reshape.reshape(-1, N//2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(-1, T, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
    
        return feature_rope.reshape(feature.shape)
    
    def forward(self, input):
        # input shape: B, N, T

        B, _, T = input.shape

        weight = self.weight(self.input_drop(self.input_norm(input))).reshape(B, self.num_head, self.hidden_size*3, T).mT
        Q, K, V = torch.split(weight, self.hidden_size, dim=-1)  # B, num_head, T, N
        
        # rotary positional embedding
        Q_rot = self._add_rotary_sequence(Q)
        K_rot = self._add_rotary_sequence(K)

        attention_output = F.scaled_dot_product_attention(Q_rot.contiguous(), K_rot.contiguous(), V.contiguous(), dropout_p=self.attention_drop, is_causal=self.causal)  # B, num_head, T, N
        attention_output = attention_output.mT.reshape(B, -1, T)
        output = self.output(attention_output) + input

        gate, z = self.MLP(output).chunk(2, dim=1)
        output = output + self.MLP_output(F.silu(gate) * z)

        return output, (K_rot, V)
    
class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super(ConvActNorm1d, self).__init__()
        
        self.in_channel = in_channel
        self.kernel = kernel
        self.causal = causal
        if not causal:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel-1)//2, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=kernel-1, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )
        
    def forward(self, input):
        
        output = self.conv(input)
        if self.causal:
            output = output[...,:-self.kernel+1]
        return input + output

class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super(ICB, self).__init__()
        
        self.blocks = nn.Sequential(ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal)
                                    )
        
    def forward(self, input):
        
        return self.blocks(input)

class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super(BSNet, self).__init__()

        self.feature_dim = feature_dim

        self.band_net = Roformer(self.feature_dim, self.feature_dim, num_head=8, window=100, causal=False)
        self.seq_net = ICB(self.feature_dim, kernel=kernel)

    def forward(self, input):
        # input shape: B, nband, N, T

        B, nband, N, T = input.shape

        # band comm
        band_input = input.permute(0,3,2,1).reshape(B*T, -1, nband)
        band_output, _ = self.band_net(band_input)
        band_output = band_output.reshape(B, T, -1, nband).permute(0,3,2,1)

        # sequence modeling
        output = self.seq_net(band_output.reshape(B*nband, -1, T)).reshape(B, nband, -1, T)  # B, nband, N, T

        return output
    
class Apollo(BaseModel):
    def __init__(
        self, 
        sr: int,
        win: int,
        feature_dim: int,
        layer: int
    ):
        super().__init__(sample_rate=sr)
        
        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # 80 bands
        bandwidth = int(self.win / 160)
        self.band_width = [bandwidth]*79
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        #print(self.band_width, self.nband)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(RMSNorm(self.band_width[i]*2+1),
                                         nn.Conv1d(self.band_width[i]*2+1, self.feature_dim, 1))
                          )

        self.net = []
        for _ in range(layer):
            self.net.append(BSNet(self.feature_dim))
        self.net = nn.Sequential(*self.net)
        
        self.output = nn.ModuleList([])
        for i in range(self.nband):
            self.output.append(nn.Sequential(RMSNorm(self.feature_dim),
                                                 nn.Conv1d(self.feature_dim, self.band_width[i]*4, 1),
                                                 nn.GLU(dim=1)
                                                )
                                  )

    def spec_band_split(self, input):

        B, nch, nsample = input.shape
        device = input.device
        is_other_gpu = is_using_other_gpu(device.type)

        if is_other_gpu:
            input = input.cpu()
        #print("MADE IT HERE 262")
        spec = torch.stft(input.view(B*nch, nsample), n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device), return_complex=True)

        if is_other_gpu:
            spec = spec.to(device)
            input = input.to(device)
        #print("MADE IT HERE 269")
        subband_spec = []
        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(self.nband):
            this_spec = spec[:,band_idx:band_idx+self.band_width[i]]
            subband_spec.append(this_spec)  # B, BW, T
            subband_power.append((this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1))  # B, 1, T
            normalized_spec = torch.complex(this_spec.real / subband_power[-1], this_spec.imag / subband_power[-1])
                
            subband_spec_norm.append(normalized_spec)
            band_idx += self.band_width[i]
        subband_power = torch.cat(subband_power, 1)  # B, nband, T

        return subband_spec_norm, subband_power

    def feature_extractor(self, input):
        
        subband_spec_norm, subband_power = self.spec_band_split(input)
        
        # normalization and bottleneck
        subband_feature = []
        for i in range(self.nband):
            concat_spec = torch.cat([subband_spec_norm[i].real, subband_spec_norm[i].imag, torch.log(subband_power[:,i].unsqueeze(1))], 1)
            subband_feature.append(self.BN[i](concat_spec))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        return subband_feature
        
    def forward(self, input):
        B, nch, nsample = input.shape

        device = input.device
        #print(device)
        is_other_gpu = is_using_other_gpu(device.type)

        if is_other_gpu:
            input = input.cpu()

        subband_feature = self.feature_extractor(input)
        feature = self.net(subband_feature)

        est_spec = []
        for i in range(self.nband):
            this_RI = self.output[i](feature[:,i]).view(B*nch, 2, self.band_width[i], -1)

            if is_other_gpu:
                this_RI = this_RI.to(input.device)

            RI_input = torch.complex(this_RI[:,0], this_RI[:,1])
            est_spec.append(RI_input)
        est_spec = torch.cat(est_spec, 1)
        output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride, 
                             window=torch.hann_window(self.win).to(input.device), length=nsample).view(B, nch, -1)
        
        #print("MADE IT HERE")
        if is_other_gpu:
            output = output.to(device)

        return output
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args