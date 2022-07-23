import torch
from torch._C import has_mkl
import torch.nn as nn
import numpy as np
import librosa

dim_c = 4
model_path = 'model'

class Conv_TDF_net_trim(nn.Module):
    def __init__(self, device, n_fft_scale, dim_f, load, model_name, target_name, 
                 L, dim_t, hop=1024):
        
        super(Conv_TDF_net_trim, self).__init__()
        
        self.dim_f, self.dim_t = dim_f, 2**dim_t
        self.n_fft = n_fft_scale
        self.hop = hop
        self.n_bins = self.n_fft//2+1
        self.chunk_size = hop * (self.dim_t-1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=False).to(device)
        self.target_name = target_name
        #print(n_fft_scale)
        out_c = dim_c*4 if target_name=='*' else dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins-self.dim_f, self.dim_t]).to(device)
        
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

def spec_effects(wave, algorithm='Default', value=None):
    doubleout = spec = [stft(wave[0],2048,1024),stft(wave[1],2048,1024)]
    if algorithm == 'Min_Mag':
        doubleout
        v_spec_m = np.where(np.abs(spec[1]) <= np.abs(spec[0]), spec[1], spec[0])
        wave = istft(v_spec_m,1024)
    elif algorithm == 'Max_Mag':
        doubleout
        v_spec_m = np.where(np.abs(spec[1]) >= np.abs(spec[0]), spec[1], spec[0])
        wave = istft(v_spec_m,1024)
    elif algorithm == 'Default':
        doubleout
        #wave = [istft(spec[0],1024),istft(spec[1],1024)]
        wave = (wave[1] * value) + (wave[0] * (1-value))
    elif algorithm == 'Invert_p':
            doubleout
            X_mag = np.abs(spec[0])
            y_mag = np.abs(spec[1])            
            max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
            v_spec = spec[1] - max_mag * np.exp(1.j * np.angle(spec[0]))
            wave = istft(v_spec,1024)
    return wave

    
def get_models(name, device, n_fft_scale, dim_f, load=True, stems='bdov'):
    
    if name=='tdf_extra':
        models = []
        if 'b' in stems:
            models.append(
                Conv_TDF_net_trim(
                    device=device, load=load, n_fft_scale=n_fft_scale,
                    model_name='Conv-TDF', target_name='bass',  
                    L=11, dim_f=dim_f, dim_t=8
                )
            )
        if 'd' in stems:
            models.append(
                Conv_TDF_net_trim(
                    device=device, load=load, n_fft_scale=n_fft_scale,
                    model_name='Conv-TDF', target_name='drums',  
                    L=9, dim_f=dim_f, dim_t=7
                )
            )
        if 'o' in stems:
            models.append(
                Conv_TDF_net_trim( 
                    device=device, load=load, n_fft_scale=n_fft_scale,
                    model_name='Conv-TDF', target_name='other',  
                    L=11, dim_f=dim_f, dim_t=8
                )
            )
        if 'v' in stems:
            models.append(
                Conv_TDF_net_trim(   
                    device=device, load=load, n_fft_scale=n_fft_scale,
                    model_name='Conv-TDF', target_name='vocals', 
                    L=11, dim_f=dim_f, dim_t=8
                )
            )
            
        return models
    
    else:
        print('Model undefined')
        return None
    


