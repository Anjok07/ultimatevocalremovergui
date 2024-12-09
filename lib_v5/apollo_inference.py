import torch
import librosa
import lib_v5.apollo_model_data as models
from tqdm.auto import tqdm
import numpy as np
from gui_data.constants import *
from separate import (get_gpu_info, clear_gpu_cache,
    cuda_available, directml_available, mps_available
)

import warnings
warnings.filterwarnings("ignore")

if not is_macos:
    import torch_directml # type:ignore

DIRECTML_DEVICE, directml_available = get_gpu_info()
is_choose_arch = cuda_available and directml_available
is_directml_only = not cuda_available and directml_available
is_cuda_only = cuda_available and not directml_available
is_gpu_available = cuda_available or directml_available or mps_available

def load_audio(file_path):
    audio, samplerate = librosa.load(file_path, mono=False, sr=44100)
    #print(f'INPUT audio.shape = {audio.shape} | samplerate = {samplerate}')
    #audio = dBgain(audio, -6)
    return torch.from_numpy(audio), samplerate

def _getWindowingArray(window_size, fade_size):
    # IMPORTANT NOTE :
    # no fades here in the end, only removing the failed ending of the chunk
    fadein = torch.linspace(1, 1, fade_size)
    fadeout = torch.linspace(0, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def dBgain(audio, volume_gain_dB):
    gain = 10 ** (volume_gain_dB / 20)
    gained_audio = audio * gain 
    return gained_audio

def check_gpu_availability(is_gpu_conversion, device_set, is_use_directml):
    device = CPU
    is_other_gpu = False
    #is_using_directml = False
    
    if is_gpu_conversion >= 0:
        if mps_available:
            device, is_other_gpu = MPS_DEVICE, True
        else:
            device_prefix = None
            if device_set != DEFAULT:
                device_prefix = DIRECTML_DEVICE if is_use_directml and directml_available else CUDA_DEVICE

            if directml_available and is_use_directml:
                device = torch_directml.device() if not device_prefix else f'{device_prefix}:{device_set}'
                is_other_gpu = True
                #is_using_directml = True
            elif cuda_available and not is_use_directml:
                device = CUDA_DEVICE if not device_prefix else f'{device_prefix}:{device_set}'
                
    return device, is_other_gpu
            
def restore_process(input_wav, ckpt_path, overlap=2, chunk_size=10, set_progress_bar=None, is_gpu_conversion=0, device_set=DEFAULT, is_use_directml=False, extracted_params=None, config=None):
    
    device, is_other_gpu = check_gpu_availability(is_gpu_conversion, device_set, is_use_directml)

    global progress_value
    progress_value = 0

    def process_chunk(chunk):
        chunk = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            return model(chunk).squeeze(0).squeeze(0).cpu()

    def progress_bar_ui(length):
        global progress_value
        progress_value += 1

        # Avoid division by zero
        if length <= 0:
            length = 1

        iter_val = (0.90 / length * progress_value)
        iter_val = 0.99 if iter_val >= 1.0 else iter_val
        set_progress_bar(0.1, iter_val)

    model = models.BaseModel.from_pretrain(ckpt_path, **extracted_params).to(device)

    audio_data, samplerate = load_audio(input_wav)
    
    C = chunk_size * samplerate  # chunk_size seconds to samples
    N = overlap
    
    step = C // N if overlap else C
    step_ui = int(step)
    
    fade_sec = 3 if chunk_size >= 3 else chunk_size

    fade_size = fade_sec * 44100 # 3 seconds
    border = C - step
    
    # handle mono inputs correctly
    if len(audio_data.shape) == 1:
        audio_data = audio_data.unsqueeze(0) 

    # Pad the input if necessary
    if audio_data.shape[1] > 2 * border and (border > 0):
        audio_data = torch.nn.functional.pad(audio_data, (border, border), mode='reflect')

    windowingArray = _getWindowingArray(C, fade_size)

    result = torch.zeros((1,) + tuple(audio_data.shape), dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(audio_data.shape), dtype=torch.float32)

    i = 0

    batch_len = max(1, int(audio_data.shape[1] / step_ui))  # Ensure batch_len is at least 1

    while i < audio_data.shape[1]:
        part = audio_data[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
            else:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

        out = process_chunk(part)

        window = windowingArray
        if i == 0:  # First audio chunk, no fadein
            window[:fade_size] = 1
        elif i + C >= audio_data.shape[1]:  # Last audio chunk, no fadeout
            window[-fade_size:] = 1

        result[..., i:i+length] += out[..., :length] * window[..., :length]
        counter[..., i:i+length] += window[..., :length]

        i += step
        
        if set_progress_bar:
            progress_bar_ui(batch_len)

    final_output = result / counter
    final_output = final_output.squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    # Remove padding if added earlier
    if audio_data.shape[1] > 2 * border and (border > 0):
        final_output = final_output[..., border:-border]

    # Memory clearing
    model.cpu()
    del model
    clear_gpu_cache()
    
    return final_output