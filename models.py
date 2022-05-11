import torch
from torch._C import has_mkl
import torch.nn as nn
import numpy as np
import librosa

dim_c = 4
k = 3
model_path = 'model'
n_fft_scale = {'bass': 8, 'drums':2, 'other':4, 'vocals':3, '*':2}


class Conv_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, bias=True):
        
        super(Conv_TDF, self).__init__()
        
        self.use_tdf = bn is not None
   
        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k//2),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )
            )

        if self.use_tdf:
            if bn==0:
                self.tdf = nn.Sequential(
                    nn.Linear(f,f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f,f//bn, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Linear(f//bn,f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU()
                )
                       
    def forward(self, x):
        for h in self.H:
            x = h(x)
        
        return x + self.tdf(x) if self.use_tdf else x


class Conv_TDF_net_trim(nn.Module):
    def __init__(self, device, load, model_name, target_name, lr, epoch, 
                 L, l, g, dim_f, dim_t, k=3, hop=1024, bn=None, bias=True):
        
        super(Conv_TDF_net_trim, self).__init__()
        
        self.dim_f, self.dim_t = 2**dim_f, 2**dim_t
        self.n_fft = self.dim_f * n_fft_scale[target_name]
        self.hop = hop
        self.n_bins = self.n_fft//2+1
        self.chunk_size = hop * (self.dim_t-1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.target_name = target_name
        self.blender = 'blender' in model_name
        
        out_c = dim_c*4 if target_name=='*' else dim_c
        in_c = dim_c*2 if self.blender else dim_c
        #out_c = dim_c*2 if self.blender else dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins-self.dim_f, self.dim_t]).to(device)
  
        self.n = L//2
        if load:
            
            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=g, kernel_size=1, stride=1),
                nn.BatchNorm2d(g),
                nn.ReLU(),
            )

            f = self.dim_f
            c = g
            self.ds_dense = nn.ModuleList()
            self.ds = nn.ModuleList()
            for i in range(self.n):
                self.ds_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))

                scale = (2,2)
                self.ds.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=c, out_channels=c+g, kernel_size=scale, stride=scale),
                        nn.BatchNorm2d(c+g),
                        nn.ReLU()
                    )
                )
                f = f//2
                c += g

            self.mid_dense = Conv_TDF(c, l, f, k, bn, bias=bias)
            #if bn is None and mid_tdf:
            #    self.mid_dense = Conv_TDF(c, l, f, k, bn=0, bias=False)

            self.us_dense = nn.ModuleList()
            self.us = nn.ModuleList()
            for i in range(self.n):
                scale = (2,2)
                self.us.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=c, out_channels=c-g, kernel_size=scale, stride=scale),
                        nn.BatchNorm2d(c-g),
                        nn.ReLU()
                    )
                )
                f = f*2
                c -= g

                self.us_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))

            
            self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=out_c, kernel_size=1, stride=1),
            )


            model_cfg = f'L{L}l{l}g{g}' 
            model_cfg += ', ' if (bn is None or bn==0) else f'bn{bn}, '

            stft_cfg = f'f{dim_f}t{dim_t}, '

            model_name = model_name[:model_name.index('(')+1] + model_cfg + stft_cfg + model_name[model_name.index('(')+1:]
            try:
                self.load_state_dict(
                    torch.load('{0}/{1}/{2}_lr{3}_e{4:05}.ckpt'.format(model_path, model_name, target_name, lr, epoch), map_location=device)
                )
                print(f'Loading model ({target_name})')
            except FileNotFoundError:
                print(f'Random init ({target_name})') 

        
    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,dim_c,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1,c,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,c,self.chunk_size])
        
    
    def forward(self, x):
        
        x = self.first_conv(x)
        
        x = x.transpose(-1,-2)
        
        ds_outputs = []
        for i in range(self.n):
            x = self.ds_dense[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)
        
        x = self.mid_dense(x)
        
        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i-1]
            x = self.us_dense[i](x)
        
        x = x.transpose(-1,-2)
        
        x = self.final_conv(x)
       
        return x
    
def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])

    return spec

def istft(spec, hl):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hl)
    wave_right = librosa.istft(spec_right, hop_length=hl)
    wave = np.asfortranarray([wave_left, wave_right])

    return wave

def spec_effects(wave, algorithm='default', value=None):
    spec = [stft(wave[0],2048,1024),stft(wave[1],2048,1024)]
    if algorithm == 'min_mag':
        v_spec_m = np.where(np.abs(spec[1]) <= np.abs(spec[0]), spec[1], spec[0])
        wave = istft(v_spec_m,1024)
    elif algorithm == 'max_mag':
        v_spec_m = np.where(np.abs(spec[1]) >= np.abs(spec[0]), spec[1], spec[0])
        wave = istft(v_spec_m,1024)
    elif algorithm == 'default':
        #wave = [istft(spec[0],1024),istft(spec[1],1024)]
        wave = (wave[1] * value) + (wave[0] * (1-value))
    elif algorithm == 'invert_p':
            X_mag = np.abs(spec[0])
            y_mag = np.abs(spec[1])            
            max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
            v_spec = spec[1] - max_mag * np.exp(1.j * np.angle(spec[0]))
            wave = istft(v_spec,1024)
    return wave

    
def get_models(name, device, load=True, stems='vocals'):
    
    if name=='tdf_extra':
        models = []
        if 'vocals' in stems:
            models.append(
                Conv_TDF_net_trim(   
                    device=device, load=load,
                    model_name='Conv-TDF', target_name='vocals', 
                    lr=0.0001, epoch=0, 
                    L=11, l=3, g=32, bn=8, bias=False, 
                    dim_f=11, dim_t=8
                )
            )
        return models
    
    else:
        print('Model undefined')
        return None
    


