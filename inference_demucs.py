from datetime import datetime
from demucs.apply import BagOfModels, apply_model
from demucs.hdemucs import HDemucs
from demucs.model_v2 import Demucs
from demucs.pretrained import get_model as _gm
from demucs.tasnet_v2 import ConvTasNet
from demucs.utils import apply_model_v1
from demucs.utils import apply_model_v2
from diffq import DiffQuantizer
from lib_v5 import spec_utils
from lib_v5.model_param_init import ModelParameters
from pathlib import Path
from random import randrange
from tqdm import tqdm
import gzip
import io
import librosa
import numpy as np
import os
import os
import os.path
import psutil
import pydub
import shutil
import soundfile as sf
import sys
import time
import time  # Timer
import tkinter as tk
import torch
import torch.hub
import traceback  # Error Message Recent Calls
import warnings
import zlib
    
class Predictor():        
    def __init__(self):
        pass
    
    def prediction_setup(self):
        
        global device

        if data['gpu'] >= 0:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
        if data['gpu'] == -1:
            device = torch.device('cpu')
        
        if demucs_model_version == 'v1':
            load_from = "models/Demucs_Models/"f"{demucs_model_set_name}"
            if str(load_from).endswith(".gz"):
                load_from = gzip.open(load_from, "rb")
            klass, args, kwargs, state = torch.load(load_from)
            self.demucs = klass(*args, **kwargs)
            widget_text.write(base_text + 'Loading Demucs v1 model... ')
            update_progress(**progress_kwargs,
            step=0.05)   
            self.demucs.to(device) 
            self.demucs.load_state_dict(state)
            widget_text.write('Done!\n')
            if not data['segment'] == 'None':
                widget_text.write(base_text + 'Segments is only available in Demucs v3. Please use \"Chunks\" instead.\n')
            else:
                pass
                
        if demucs_model_version == 'v2':
            if '48' in demucs_model_set_name:
                channels=48
            elif 'unittest' in demucs_model_set_name:
                channels=4
            else:
                channels=64
                
            if 'tasnet' in demucs_model_set_name:
                self.demucs = ConvTasNet(sources=["drums", "bass", "other", "vocals"], X=10)
            else:
                self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=channels)
            widget_text.write(base_text + 'Loading Demucs v2 model... ')
            update_progress(**progress_kwargs,
            step=0.05)   
            self.demucs.to(device) 
            self.demucs.load_state_dict(torch.load("models/Demucs_Models/"f"{demucs_model_set_name}"))
            widget_text.write('Done!\n')
            if not data['segment'] == 'None':
                widget_text.write(base_text + 'Segments is only available in Demucs v3. Please use \"Chunks\" instead.\n')
            else:
                pass
            self.demucs.eval()
            
        if demucs_model_version == 'v3':
            self.demucs = HDemucs(sources=["drums", "bass", "other", "vocals"])
            widget_text.write(base_text + 'Loading Demucs model... ')
            update_progress(**progress_kwargs,
            step=0.05)   
            path_d = Path('models/Demucs_Models/v3_repo')
            #print('What Demucs model was chosen? ', demucs_model_set_name)
            self.demucs = _gm(name=demucs_model_set_name, repo=path_d)
            widget_text.write('Done!\n')
            if 'UVR' in data['DemucsModel']:
                widget_text.write(base_text + "2 stem model selected.\n")
            if isinstance(self.demucs, BagOfModels):
                widget_text.write(base_text + f"Selected model is a bag of {len(self.demucs.models)} models.\n") 
            
            if data['segment'] == 'None':
                segment = None
                if isinstance(self.demucs, BagOfModels):
                    if segment is not None:
                        for sub in self.demucs.models:
                            sub.segment = segment
                else:
                    if segment is not None:
                        sub.segment = segment
            else:
                try:
                    segment = int(data['segment'])
                    if isinstance(self.demucs, BagOfModels):
                        if segment is not None:
                            for sub in self.demucs.models:
                                sub.segment = segment
                    else:
                        if segment is not None:
                            sub.segment = segment
                    widget_text.write(base_text + "Segments set to "f"{segment}.\n")
                except:
                    segment = None
                    if isinstance(self.demucs, BagOfModels):
                        if segment is not None:
                            for sub in self.demucs.models:
                                sub.segment = segment
                    else:
                        if segment is not None:
                            sub.segment = segment
                    
                self.demucs.to(device)
                self.demucs.eval()
            
        update_progress(**progress_kwargs,
        step=0.1)
        
    def prediction(self, m):  

        mix, samplerate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix,mix])
        
        mix = mix.T
        sources = self.demix(mix.T)
        widget_text.write(base_text + 'Inferences complete!\n')
    
        #Main Save Path
        save_path = os.path.dirname(_basename)
        
        vocals_name = '(Vocals)'
        other_name = '(Other)'
        drums_name = '(Drums)'
        bass_name = '(Bass)'
              
        vocals_path = '{save_path}/{file_name}.wav'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{vocals_name}',)
        vocals_path_mp3 = '{save_path}/{file_name}.mp3'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{vocals_name}',)
        vocals_path_flac = '{save_path}/{file_name}.flac'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{vocals_name}',)
        
        #Other
        
        other_path = '{save_path}/{file_name}.wav'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{other_name}',)
        other_path_mp3 = '{save_path}/{file_name}.mp3'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{other_name}',)
        other_path_flac = '{save_path}/{file_name}.flac'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{other_name}',)
            
        #Drums
        
        drums_path = '{save_path}/{file_name}.wav'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{drums_name}',)
        drums_path_mp3 = '{save_path}/{file_name}.mp3'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{drums_name}',)
        drums_path_flac = '{save_path}/{file_name}.flac'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{drums_name}',)
        
        #Bass
        

        bass_path = '{save_path}/{file_name}.wav'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{bass_name}',)
        bass_path_mp3 = '{save_path}/{file_name}.mp3'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{bass_name}',)
        bass_path_flac = '{save_path}/{file_name}.flac'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{bass_name}',)
                
        
                
        #If not 'All Stems'

        if stemset_n == '(Vocals)':
            vocal_name = '(Vocals)'
        elif stemset_n == '(Other)':
            vocal_name = '(Other)'
        elif stemset_n == '(Drums)':
            vocal_name = '(Drums)'
        elif stemset_n == '(Bass)':
            vocal_name = '(Bass)'
        elif stemset_n == '(Instrumental)':
            vocal_name = '(Instrumental)'
            
        vocal_path = '{save_path}/{file_name}.wav'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{vocal_name}',)
        vocal_path_mp3 = '{save_path}/{file_name}.mp3'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{vocal_name}',)
        vocal_path_flac = '{save_path}/{file_name}.flac'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{vocal_name}',)
        
        #Instrumental Path

        if stemset_n == '(Vocals)':
            Instrumental_name = '(Instrumental)'
        elif stemset_n == '(Other)':
            Instrumental_name = '(No_Other)'
        elif stemset_n == '(Drums)':
            Instrumental_name = '(No_Drums)'
        elif stemset_n == '(Bass)':
            Instrumental_name = '(No_Bass)'
        elif stemset_n == '(Instrumental)':
            if data['demucs_stems'] == 'All Stems':
                Instrumental_name = '(Instrumental)'
            else:
                Instrumental_name = '(Vocals)'
                

        Instrumental_path = '{save_path}/{file_name}.wav'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{Instrumental_name}',)
        Instrumental_path_mp3 = '{save_path}/{file_name}.mp3'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{Instrumental_name}',)
        Instrumental_path_flac = '{save_path}/{file_name}.flac'.format(
            save_path=save_path,
            file_name = f'{os.path.basename(_basename)}_{Instrumental_name}',)   

            
        if os.path.isfile(vocal_path):
            file_exists_n = 'there'
        else:
            file_exists_n = 'not_there'

        if os.path.isfile(vocal_path):
            file_exists_v = 'there'
        else:
            file_exists_v = 'not_there'
            
        if os.path.isfile(Instrumental_path):
            file_exists_i = 'there'
        else:
            file_exists_i = 'not_there'


        if not data['demucs_stems'] == 'All Stems':
            if data['inst_only_b']:
                widget_text.write(base_text + 'Preparing mixture without selected stem...')
            else:
                widget_text.write(base_text + 'Saving Stem(s)... ')
        else:
            pass
            
        if data['demucs_stems'] == 'All Stems':  
            
            if data['saveFormat'] == 'Wav':
                widget_text.write(base_text + 'Saving Stem(s)... ')
            else:
                pass

            if 'UVR' in model_set_name:
                sf.write(Instrumental_path, normalization_set(sources[0]).T, samplerate, subtype=wav_type_set)
                sf.write(vocals_path, normalization_set(sources[1]).T, samplerate, subtype=wav_type_set)
            else:
                sf.write(bass_path, normalization_set(sources[0]).T, samplerate, subtype=wav_type_set)
                sf.write(drums_path, normalization_set(sources[1]).T, samplerate, subtype=wav_type_set)
                sf.write(other_path, normalization_set(sources[2]).T, samplerate, subtype=wav_type_set)
                sf.write(vocals_path, normalization_set(sources[3]).T, samplerate, subtype=wav_type_set)
            
            if data['saveFormat'] == 'Mp3':
                try:
                    if 'UVR' in model_set_name:
                        widget_text.write(base_text + 'Saving Stem(s) as Mp3(s)... ')
                        musfile = pydub.AudioSegment.from_wav(vocals_path)
                        musfile.export(vocals_path_mp3, format="mp3", bitrate=mp3_bit_set)    
                        musfile = pydub.AudioSegment.from_wav(Instrumental_path)
                        musfile.export(Instrumental_path_mp3, format="mp3", bitrate=mp3_bit_set) 
                        try:
                            os.remove(Instrumental_path)
                            os.remove(vocals_path)
                        except:
                            pass
                    else:
                        widget_text.write(base_text + 'Saving Stem(s) as Mp3(s)... ')
                        musfile = pydub.AudioSegment.from_wav(drums_path)
                        musfile.export(drums_path_mp3, format="mp3", bitrate=mp3_bit_set)    
                        musfile = pydub.AudioSegment.from_wav(bass_path)
                        musfile.export(bass_path_mp3, format="mp3", bitrate=mp3_bit_set)   
                        musfile = pydub.AudioSegment.from_wav(other_path)
                        musfile.export(other_path_mp3, format="mp3", bitrate=mp3_bit_set)   
                        musfile = pydub.AudioSegment.from_wav(vocals_path)
                        musfile.export(vocals_path_mp3, format="mp3", bitrate=mp3_bit_set)  
                        try:
                            os.remove(drums_path)
                            os.remove(bass_path)
                            os.remove(other_path)
                            os.remove(vocals_path)
                        except:
                            pass
                except Exception as e:
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                    if "ffmpeg" in errmessage:
                        widget_text.write('\n' + base_text + 'Failed to save output(s) as Mp3(s).\n')
                        widget_text.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    else:
                        widget_text.write(base_text + 'Failed to save output(s) as Mp3(s).\n')
                        widget_text.write(base_text + 'Please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    try:
                        with open('errorlog.txt', 'w') as f:
                            f.write(f'Last Error Received:\n\n' +
                                    f'Error Received while attempting to save file as mp3 "{os.path.basename(music_file)}":\n\n' + 
                                    f'Process Method: Demucs v3\n\n' +
                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                    f'If this error persists, please contact the developers.\n\n' + 
                                    f'Raw error details:\n\n' +
                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                    except:
                        pass   
            elif data['saveFormat'] == 'Flac':
                try:
                    if 'UVR' in model_set_name:
                        widget_text.write(base_text + 'Saving Stem(s) as flac(s)... ')
                        musfile = pydub.AudioSegment.from_wav(vocals_path)
                        musfile.export(vocals_path_flac, format="flac")    
                        musfile = pydub.AudioSegment.from_wav(Instrumental_path)
                        musfile.export(Instrumental_path_flac, format="flac") 
                        try:
                            os.remove(Instrumental_path)
                            os.remove(vocals_path)
                        except:
                            pass
                    else:
                        widget_text.write(base_text + 'Saving Stem(s) as Flac(s)... ')
                        musfile = pydub.AudioSegment.from_wav(drums_path)
                        musfile.export(drums_path_flac, format="flac")    
                        musfile = pydub.AudioSegment.from_wav(bass_path)
                        musfile.export(bass_path_flac, format="flac")   
                        musfile = pydub.AudioSegment.from_wav(other_path)
                        musfile.export(other_path_flac, format="flac")   
                        musfile = pydub.AudioSegment.from_wav(vocals_path)
                        musfile.export(vocals_path_flac, format="flac")  
                        try:
                            os.remove(drums_path)
                            os.remove(bass_path)
                            os.remove(other_path)
                            os.remove(vocals_path)
                        except:
                            pass
                except Exception as e:
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                    if "ffmpeg" in errmessage:
                        widget_text.write('\n' + base_text + 'Failed to save output(s) as Flac(s).\n')
                        widget_text.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    else:
                        widget_text.write(base_text + 'Failed to save output(s) as flac(s).\n')
                        widget_text.write(base_text + 'Please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    try:
                        with open('errorlog.txt', 'w') as f:
                            f.write(f'Last Error Received:\n\n' +
                                    f'Error Received while attempting to save file as flac "{os.path.basename(music_file)}":\n\n' + 
                                    f'Process Method: Demucs v3\n\n' +
                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                    f'If this error persists, please contact the developers.\n\n' + 
                                    f'Raw error details:\n\n' +
                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                    except:
                        pass
            elif data['saveFormat'] == 'Wav':
                pass
            
            widget_text.write('Done!\n')
        else:
            if 'UVR' in model_set_name:
                if stemset_n == '(Vocals)':
                    sf.write(vocal_path, sources[1].T, samplerate, subtype=wav_type_set)
                else:
                    sf.write(vocal_path, sources[source_val].T, samplerate, subtype=wav_type_set)
            else:
                sf.write(vocal_path, sources[source_val].T, samplerate, subtype=wav_type_set)
                
            widget_text.write('Done!\n')
        
        update_progress(**progress_kwargs,
        step=(0.9))
        
        if data['demucs_stems'] == 'All Stems':
            pass
        else:
            if data['voc_only_b'] and not data['inst_only_b']:
                pass

            else:
                finalfiles = [
                    {
                        'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                        'files':[str(music_file), vocal_path],
                    }
                ]         
                widget_text.write(base_text + 'Saving Instrumental... ')      
                for i, e in tqdm(enumerate(finalfiles)):

                    wave, specs = {}, {}
                            
                    mp = ModelParameters(e['model_params'])
                    
                    for i in range(len(e['files'])):    
                        spec = {}
                        
                        for d in range(len(mp.param['band']), 0, -1):          
                            bp = mp.param['band'][d]            
                            
                            if d == len(mp.param['band']): # high-end band                
                                wave[d], _ = librosa.load(
                                    e['files'][i], bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                                
                                if len(wave[d].shape) == 1: # mono to stereo
                                    wave[d] = np.array([wave[d], wave[d]])
                            else: # lower bands
                                wave[d] = librosa.resample(wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
                                    
                            spec[d] = spec_utils.wave_to_spectrogram(wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['mid_side_b2'], mp.param['reverse'])
                            
                        specs[i] = spec_utils.combine_spectrograms(spec, mp)
                    
                    del wave   
                    
                    ln = min([specs[0].shape[2], specs[1].shape[2]])
                    specs[0] = specs[0][:,:,:ln]
                    specs[1] = specs[1][:,:,:ln]
                    X_mag = np.abs(specs[0])
                    y_mag = np.abs(specs[1])            
                    max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
                    v_spec = specs[1] - max_mag * np.exp(1.j * np.angle(specs[0]))
                    update_progress(**progress_kwargs,
                    step=(1))
                    
                    sf.write(Instrumental_path, normalization_set(spec_utils.cmb_spectrogram_to_wave(-v_spec, mp)), mp.param['sr'], subtype=wav_type_set)
                
                    
                if data['inst_only_b']:
                    if file_exists_v == 'there':
                        pass
                    else:
                        try:
                            os.remove(vocal_path)
                        except:
                            pass
     
                widget_text.write('Done!\n')
          
        
        if not data['demucs_stems'] == 'All Stems':
            
            if data['saveFormat'] == 'Mp3':
                try:
                    if data['inst_only_b'] == True:
                        pass
                    else:
                        musfile = pydub.AudioSegment.from_wav(vocal_path)
                        musfile.export(vocal_path_mp3, format="mp3", bitrate=mp3_bit_set)
                        if file_exists_v == 'there':
                            pass
                        else:
                            try:
                                os.remove(vocal_path)
                            except:
                                pass
                    if data['voc_only_b'] == True:
                        pass
                    else:
                        musfile = pydub.AudioSegment.from_wav(Instrumental_path)
                        musfile.export(Instrumental_path_mp3, format="mp3", bitrate=mp3_bit_set)
                        if file_exists_i == 'there':
                            pass
                        else:
                            try:
                                os.remove(Instrumental_path)
                            except:
                                pass

                            
                except Exception as e:
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                    if "ffmpeg" in errmessage:
                        widget_text.write(base_text + 'Failed to save output(s) as Mp3(s).\n')
                        widget_text.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    else:
                        widget_text.write(base_text + 'Failed to save output(s) as Mp3(s).\n')
                        widget_text.write(base_text + 'Please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    try:
                        with open('errorlog.txt', 'w') as f:
                            f.write(f'Last Error Received:\n\n' +
                                    f'Error Received while attempting to save file as mp3 "{os.path.basename(music_file)}":\n\n' + 
                                    f'Process Method: Demucs v3\n\n' +
                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                    f'If this error persists, please contact the developers.\n\n' + 
                                    f'Raw error details:\n\n' +
                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                    except:
                        pass
                
            if data['saveFormat'] == 'Flac':
                try:
                    if data['inst_only_b'] == True:
                        pass
                    else:
                        musfile = pydub.AudioSegment.from_wav(vocal_path)
                        musfile.export(vocal_path_flac, format="flac") 
                        if file_exists_v == 'there':
                            pass
                        else:
                            try:
                                os.remove(vocal_path)
                            except:
                                pass
                    if data['voc_only_b'] == True:
                        pass
                    else:
                        musfile = pydub.AudioSegment.from_wav(Instrumental_path)
                        musfile.export(Instrumental_path_flac, format="flac")  
                        if file_exists_i == 'there':
                            pass
                        else:
                            try:
                                os.remove(Instrumental_path)
                            except:
                                pass  
                            
                except Exception as e:
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                    if "ffmpeg" in errmessage:
                        widget_text.write(base_text + 'Failed to save output(s) as Flac(s).\n')
                        widget_text.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    else:
                        widget_text.write(base_text + 'Failed to save output(s) as Flac(s).\n')
                        widget_text.write(base_text + 'Please check error log.\n')
                        widget_text.write(base_text + 'Moving on...\n')
                    try:
                        with open('errorlog.txt', 'w') as f:
                            f.write(f'Last Error Received:\n\n' +
                                    f'Error Received while attempting to save file as flac "{os.path.basename(music_file)}":\n\n' + 
                                    f'Process Method: Demucs v3\n\n' +
                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                    f'If this error persists, please contact the developers.\n\n' + 
                                    f'Raw error details:\n\n' +
                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                    except:
                        pass
            

                if data['inst_only_b']:
                    if file_exists_n == 'there':
                        pass
                    else:
                        try:
                            os.remove(vocal_path)
                        except:
                            pass
                else:
                    try:
                        os.remove(vocal_path)
                    except:
                        pass
        
        widget_text.write(base_text + 'Completed Separation!\n')

    def demix(self, mix):

        # 1 = demucs only
        # 0 = onnx only
        if data['chunks_d'] == 'Full':
            if split_mode == True:
                chunk_set = 0
            else:
                widget_text.write(base_text + "Chunk size set to full... \n")
                chunk_set = 0
        else: 
            chunk_set = data['chunks']

        if data['chunks_d'] == 'Auto':
            if split_mode == True:
                widget_text.write(base_text + "Split Mode is on (Chunks disabled).\n")
                chunk_set = 0
            else:
                widget_text.write(base_text + "Split Mode is off (Chunks enabled).\n")
                if data['gpu'] == 0:
                    try:
                        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory/1.074e+9)
                    except:
                        widget_text.write(base_text + 'NVIDIA GPU Required for conversion!\n')
                    if int(gpu_mem) <= int(6):
                        chunk_set = int(10)
                        widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                    if gpu_mem in [7, 8, 9]:
                        chunk_set = int(30)
                        widget_text.write(base_text + 'Chunk size auto-set to 30... \n')
                    if gpu_mem in [10, 11, 12, 13, 14, 15]:
                        chunk_set = int(50)
                        widget_text.write(base_text + 'Chunk size auto-set to 50... \n')
                    if int(gpu_mem) >= int(16):
                        chunk_set = int(0)
                        widget_text.write(base_text + 'Chunk size auto-set to Full... \n')
                if data['gpu'] == -1:
                    sys_mem = psutil.virtual_memory().total >> 30
                    if int(sys_mem) <= int(4):
                        chunk_set = int(5)
                        widget_text.write(base_text + 'Chunk size auto-set to 5... \n')
                    if sys_mem in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                        chunk_set = int(10)
                        widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                    if sys_mem in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32]:
                        chunk_set = int(40)
                        widget_text.write(base_text + 'Chunk size auto-set to 40... \n')
                    if int(sys_mem) >= int(33):
                        chunk_set = int(0)
                        widget_text.write(base_text + 'Chunk size auto-set to Full... \n')
        else:
            if split_mode == True:
                widget_text.write(base_text + "Split Mode is on (Chunks disabled).\n")
                chunk_set = 0
            else:
                widget_text.write(base_text + "Split Mode is off (Chunks enabled).\n")
                chunk_set = int(data['chunks_d'])
                widget_text.write(base_text + "Chunk size user-set to "f"{chunk_set}... \n")
            
        samples = mix.shape[-1]
        margin = margin_set
        chunk_size = chunk_set*44100
        assert not margin == 0, 'margin cannot be zero!'
        if margin > chunk_size:
            margin = chunk_size


        segmented_mix = {}
        
        if chunk_set == 0 or samples < chunk_size:
            chunk_size = samples
        
        counter = -1
        for skip in range(0, samples, chunk_size):
            counter+=1
    
            s_margin = 0 if counter == 0 else margin
            end = min(skip+chunk_size+margin, samples)

            start = skip-s_margin

            segmented_mix[skip] = mix[:,start:end].copy()
            if end == samples:
                break
        
        if demucs_model_version == 'v1':
            sources = self.demix_demucs_v1(segmented_mix, margin_size=margin)
        if demucs_model_version == 'v2':
            sources = self.demix_demucs_v2(segmented_mix, margin_size=margin)
        if demucs_model_version == 'v3':
            sources = self.demix_demucs(segmented_mix, margin_size=margin)

        return sources
    
    def demix_demucs(self, mix, margin_size):
        
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
            
        widget_text.write(base_text + "Running Demucs Inference...\n")
        widget_text.write(base_text + "Processing "f"{len(mix)} slices... ")
        print(' Running Demucs Inference...')
        for nmix in mix:
            gui_progress_bar_demucs += 1
            update_progress(**progress_kwargs,
                step=(0.1 + (1.7/demucsitera_calc * gui_progress_bar_demucs)))
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            with torch.no_grad():
                sources = apply_model(self.demucs, cmix[None], split=split_mode, device=device, overlap=overlap_set, shifts=shift_set, progress=False)[0]
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            start = 0 if nmix == 0 else margin_size
            end = None if nmix == list(mix.keys())[::-1][0] else -margin_size
            if margin_size == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()

        sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)
        widget_text.write('Done!\n')
        return sources
    
    def demix_demucs_v1(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        widget_text.write(base_text + "Running Demucs v1 Inference...\n")
        widget_text.write(base_text + "Processing "f"{len(mix)} slices... ")
        print(' Running Demucs Inference...')
        for nmix in mix:
            gui_progress_bar_demucs += 1
            update_progress(**progress_kwargs,
                step=(0.35 + (1.05/demucsitera_calc * gui_progress_bar_demucs)))
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            with torch.no_grad():
                sources = apply_model_v1(self.demucs, cmix.to(device), split=split_mode, shifts=shift_set)
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            start = 0 if nmix == 0 else margin_size
            end = None if nmix == list(mix.keys())[::-1][0] else -margin_size
            if margin_size == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()

        sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)
        widget_text.write('Done!\n')
        return sources
    
    def demix_demucs_v2(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        widget_text.write(base_text + "Running Demucs v2 Inference...\n")
        widget_text.write(base_text + "Processing "f"{len(mix)} slices... ")
        print(' Running Demucs Inference...')
        for nmix in mix:
            gui_progress_bar_demucs += 1
            update_progress(**progress_kwargs,
                step=(0.35 + (1.05/demucsitera_calc * gui_progress_bar_demucs)))
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            shift_set = 0
            with torch.no_grad():
                sources = apply_model_v2(self.demucs, cmix.to(device), split=split_mode, overlap=overlap_set, shifts=shift_set)
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            start = 0 if nmix == 0 else margin_size
            end = None if nmix == list(mix.keys())[::-1][0] else -margin_size
            if margin_size == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()

        sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)
        widget_text.write('Done!\n')
        return sources
        
data = {
    'audfile': True,
    'chunks_d': 'Full',
    'compensate': 1.03597672895,
    'demucs_stems': 'All Stems',
    'DemucsModel': 'mdx_extra',
    'demucsmodel': True,
    'export_path': None,
    'gpu': -1,
    'input_paths': None,
    'inst_only_b': False,
    'margin': 44100,
    'mp3bit': '320k',
    'normalize': False,
    'overlap_b': 0.25,
    'saveFormat': 'Wav',
    'segment': 'None',
    'settest': False,
    'shifts_b': 2,
    'split_mode': False,
    'voc_only_b': False,
    'wavtype': 'PCM_16',
}
default_chunks = data['chunks_d']

def update_progress(progress_var, total_files, file_num, step: float = 1):
    """Calculate the progress for the progress widget in the GUI"""
    base = (100 / total_files)
    progress = base * (file_num - 1)
    progress += base * step

    progress_var.set(progress)

def get_baseText(total_files, file_num):
    """Create the base text for the command widget"""
    text = 'File {file_num}/{total_files} '.format(file_num=file_num,
                                                total_files=total_files)
    return text

warnings.filterwarnings("ignore")
cpu = torch.device('cpu')

def hide_opt():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def main(window: tk.Wm, text_widget: tk.Text, button_widget: tk.Button, progress_var: tk.Variable,
         **kwargs: dict):

    global widget_text
    global gui_progress_bar
    global music_file
    global default_chunks
    global _basename
    global _mixture
    global progress_kwargs
    global base_text
    global model_set_name
    global stemset_n
    global channel_set
    global margin_set
    global overlap_set
    global shift_set
    global source_val
    global split_mode
    global demucs_model_set_name
    global demucs_model_version
    
    global wav_type_set
    global flac_type_set
    global mp3_bit_set
    global normalization_set
    
    wav_type_set = data['wavtype']
        
    # Update default settings
    default_chunks = data['chunks_d']
    
    widget_text = text_widget
    gui_progress_bar = progress_var
    
    #Error Handling
    
    onnxmissing = "[ONNXRuntimeError] : 3 : NO_SUCHFILE"
    onnxmemerror = "onnxruntime::CudaCall CUDA failure 2: out of memory"
    onnxmemerror2 = "onnxruntime::BFCArena::AllocateRawInternal"
    systemmemerr = "DefaultCPUAllocator: not enough memory"
    runtimeerr = "CUDNN error executing cudnnSetTensorNdDescriptor"
    cuda_err = "CUDA out of memory"
    mod_err = "ModuleNotFoundError"
    file_err = "FileNotFoundError"
    ffmp_err = """audioread\__init__.py", line 116, in audio_open"""
    sf_write_err = "sf.write"
    model_adv_set_err = "Got invalid dimensions for input"
    
    try:
        with open('errorlog.txt', 'w') as f:
            f.write(f'No errors to report at this time.' + f'\n\nLast Process Method Used: MDX-Net' +
                    f'\nLast Conversion Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
    except:
        pass
    
    timestampnum = round(datetime.utcnow().timestamp())
    randomnum = randrange(100000, 1000000)
    
    data.update(kwargs)
        
    if data['wavtype'] == '32-bit Float':
        wav_type_set = 'FLOAT'
    elif data['wavtype'] == '64-bit Float':
        wav_type_set = 'DOUBLE'
    else:
        wav_type_set = data['wavtype']
        
    flac_type_set = data['flactype']
    mp3_bit_set = data['mp3bit']
    
    if data['normalize'] == True:
        normalization_set = spec_utils.normalize
        print('normalization on')
    else:
        normalization_set = spec_utils.nonormalize
        print('normalization off')
        
    stime = time.perf_counter()
    progress_var.set(0)
    text_widget.clear()
    button_widget.configure(state=tk.DISABLED)  # Disable Button
    
    
    if data['DemucsModel'] == "Tasnet v1":
        demucs_model_set_name = 'tasnet.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Tasnet_extra v1":
        demucs_model_set_name = 'tasnet_extra.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Demucs v1":
        demucs_model_set_name = 'demucs.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Demucs v1.gz":
        demucs_model_set_name = 'demucs.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Demucs_extra v1":
        demucs_model_set_name = 'demucs_extra.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Demucs_extra v1.gz":
        demucs_model_set_name = 'demucs_extra.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Light v1":
        demucs_model_set_name = 'light.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Light v1.gz":
        demucs_model_set_name = 'light.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Light_extra v1":
        demucs_model_set_name = 'light_extra.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Light_extra v1.gz":
        demucs_model_set_name = 'light_extra.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel'] == "Tasnet v2":
        demucs_model_set_name = 'tasnet-beb46fac.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel'] == "Tasnet_extra v2":
        demucs_model_set_name = 'tasnet_extra-df3777b2.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel'] == "Demucs48_hq v2":
        demucs_model_set_name = 'demucs48_hq-28a1282c.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel'] == "Demucs v2":
        demucs_model_set_name = 'demucs-e07c671f.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel'] == "Demucs_extra v2":
        demucs_model_set_name = 'demucs_extra-3646af93.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel'] == "Demucs_unittest v2":
        demucs_model_set_name = 'demucs_unittest-09ebc15f.th'
        demucs_model_version = 'v2'
    elif '.ckpt' in data['DemucsModel'] and 'v2' in data['DemucsModel']:
        demucs_model_set_name = data['DemucsModel']
        demucs_model_version = 'v2'
    elif '.ckpt' in data['DemucsModel'] and 'v1' in data['DemucsModel']:
        demucs_model_set_name = data['DemucsModel']
        demucs_model_version = 'v1'
    elif '.gz' in data['DemucsModel']:
        demucs_model_set_name = data['DemucsModel']
        demucs_model_version = 'v1'
    else:
        demucs_model_set_name = data['DemucsModel']
        demucs_model_version = 'v3'
        

    try:    #Load File(s)
        for file_num, music_file in tqdm(enumerate(data['input_paths'], start=1)):
        
            if data['wavtype'] == '64-bit Float':
                if data['saveFormat'] == 'Flac':
                    text_widget.write('Please select \"WAV\" as your save format to use 64-bit Float.\n')
                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                    progress_var.set(0)
                    button_widget.configure(state=tk.NORMAL)  # Enable Button
                    return 
                
            if data['wavtype'] == '64-bit Float':
                if data['saveFormat'] == 'Mp3':
                    text_widget.write('Please select \"WAV\" as your save format to use 64-bit Float.\n')
                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                    progress_var.set(0)
                    button_widget.configure(state=tk.NORMAL)  # Enable Button
                    return 
        
            model_set_name = data['DemucsModel']
        
            if data['demucs_stems'] == 'Vocals':
                source_val = 3
                stemset_n = '(Vocals)'
            if data['demucs_stems'] == 'Other':
                if 'UVR' in model_set_name:
                    source_val = 0
                    stemset_n = '(Instrumental)'
                else:
                    source_val = 2
                    stemset_n = '(Other)'
            if data['demucs_stems'] == 'Drums':
                if 'UVR' in model_set_name:
                    text_widget.write('You can only choose "Vocals" or "Other" stems when using this model.\n')
                    text_widget.write('Please select one of the stock Demucs models and try again.\n\n')
                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                    progress_var.set(0)
                    button_widget.configure(state=tk.NORMAL)  # Enable Button
                    return 
                else:
                    source_val = 1
                    stemset_n = '(Drums)'
            if data['demucs_stems'] == 'Bass':
                if 'UVR' in model_set_name:
                    text_widget.write('You can only choose "Vocals" or "Other" stems when using this model.\n')
                    text_widget.write('Please select one of the stock Demucs models and try again.\n\n')
                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                    progress_var.set(0)
                    button_widget.configure(state=tk.NORMAL)  # Enable Button
                    return 
                else:
                    source_val = 0
                    stemset_n = '(Bass)'
            if data['demucs_stems'] == 'All Stems':
                source_val = 3
                stemset_n = '(Instrumental)'
    
    
            overlap_set = float(data['overlap_b'])
            channel_set = int(data['channel'])
            margin_set = int(data['margin'])
            shift_set = int(data['shifts_b'])
            
            split_mode = data['split_mode']
                
            #print('Split? ', split_mode)
            
            def determinemusicfileFolderName():
                """
                Determine the name that is used for the folder and appended
                to the back of the music files
                """
                songFolderName = ''

                if str(music_file):
                    songFolderName += os.path.splitext(os.path.basename(music_file))[0]

                if songFolderName:

                    songFolderName = '/' + songFolderName


                return songFolderName
            
            def determinemodelFolderName():
                """
                Determine the name that is used for the folder and appended
                to the back of the music files
                """
                modelFolderName = ''

                if str(model_set_name):
                    modelFolderName += os.path.splitext(os.path.basename(model_set_name))[0]

                if modelFolderName:

                    modelFolderName = '/' + modelFolderName


                return modelFolderName
            
 
            if data['audfile'] == True:   
                modelFolderName = determinemodelFolderName()
                songFolderName = determinemusicfileFolderName()
                
                if modelFolderName:
                    folder_path = f'{data["export_path"]}{modelFolderName}'
                    if not os.path.isdir(folder_path):
                        os.mkdir(folder_path)
                        
                if songFolderName:
                    folder_path = f'{data["export_path"]}{modelFolderName}{songFolderName}'
                    if not os.path.isdir(folder_path):
                        os.mkdir(folder_path)
                
                _mixture = f'{data["input_paths"]}'
                if data['settest']:
                    try:
                        _basename = f'{data["export_path"]}{modelFolderName}{songFolderName}/{str(timestampnum)}_{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                    except:
                        _basename = f'{data["export_path"]}{modelFolderName}{songFolderName}/{str(randomnum)}_{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                else:
                    _basename = f'{data["export_path"]}{modelFolderName}{songFolderName}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
            else:
                _mixture = f'{data["input_paths"]}'
                if data['settest']:
                    try:
                        _basename = f'{data["export_path"]}/{str(timestampnum)}_{file_num}_{model_set_name}_{os.path.splitext(os.path.basename(music_file))[0]}'
                    except:
                        _basename = f'{data["export_path"]}/{str(randomnum)}{file_num}_{model_set_name}_{os.path.splitext(os.path.basename(music_file))[0]}'
                else:
                    _basename = f'{data["export_path"]}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                
                #if ('models/MDX_Net_Models/' + model_set + '.onnx')
                
                
            # -Get text and update progress-
            base_text = get_baseText(total_files=len(data['input_paths']),
                                        file_num=file_num)
            progress_kwargs = {'progress_var': progress_var,
                            'total_files': len(data['input_paths']),
                            'file_num': file_num}
            
            try:
                
                total, used, free = shutil.disk_usage("/") 
                
                    
                total_space = int(total/1.074e+9)
                used_space = int(used/1.074e+9)
                free_space = int(free/1.074e+9)
                
                    
                if int(free/1.074e+9) <= int(2):
                    text_widget.write('Error: Not enough storage on main drive to continue. Your main drive must have \nat least 3 GB\'s of storage in order for this application function properly. \n\nPlease ensure your main drive has at least 3 GB\'s of storage and try again.\n\n')
                    text_widget.write('Detected Total Space: ' + str(total_space) + ' GB' + '\n')
                    text_widget.write('Detected Used Space: ' + str(used_space) + ' GB' + '\n')
                    text_widget.write('Detected Free Space: ' + str(free_space) + ' GB' + '\n')
                    progress_var.set(0)
                    button_widget.configure(state=tk.NORMAL)  # Enable Button
                    return 
                
                if int(free/1.074e+9) in [3, 4, 5, 6, 7, 8]:
                    text_widget.write('Warning: Your main drive is running low on storage. Your main drive must have \nat least 3 GB\'s of storage in order for this application function properly.\n\n')
                    text_widget.write('Detected Total Space: ' + str(total_space) + ' GB' + '\n')
                    text_widget.write('Detected Used Space: ' + str(used_space) + ' GB' + '\n')
                    text_widget.write('Detected Free Space: ' + str(free_space) + ' GB' + '\n\n')
            except:
                pass
        
            update_progress(**progress_kwargs,
                            step=0)       
            
            e = os.path.join(data["export_path"])
            
            demucsmodel = 'models/Demucs_Models/' + str(data['DemucsModel'])

            pred = Predictor()
            pred.prediction_setup()
            
            # split
            pred.prediction(
                m=music_file,
            )
            
    except Exception as e:
        traceback_text = ''.join(traceback.format_tb(e.__traceback__))
        message = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
        if runtimeerr in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Your PC cannot process this audio file with the chunk size selected.\nPlease lower the chunk size and try again.\n\n')
            text_widget.write(f'If this error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'Your PC cannot process this audio file with the chunk size selected.\nPlease lower the chunk size and try again.\n\n' +
                            f'If this error persists, please contact the developers.\n\n' + 
                            f'Raw error details:\n\n' +
                            message + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if cuda_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough GPU memory to use this model.\n')
            text_widget.write(f'Please close any GPU intensive applications and try again.\n')
            text_widget.write(f'If the error persists, your GPU might not be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'The application was unable to allocate enough GPU memory to use this model.\n' + 
                            f'Please close any GPU intensive applications and try again.\n' + 
                            f'If the error persists, your GPU might not be supported.\n\n' + 
                            f'Raw error details:\n\n' +
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return
        
        if mod_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Application files(s) are missing.\n')
            text_widget.write("\n" + f'{type(e).__name__} - "{e}"' + "\n\n")
            text_widget.write(f'Please check for missing files/scripts in the app directory and try again.\n')
            text_widget.write(f'If the error persists, please reinstall application or contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'Application files(s) are missing.\n' + 
                            f'Please check for missing files/scripts in the app directory and try again.\n' + 
                            f'If the error persists, please reinstall application or contact the developers.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if file_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Missing file error raised.\n')
            text_widget.write("\n" + f'{type(e).__name__} - "{e}"' + "\n\n")
            text_widget.write("\n" + f'Please address the error and try again.' + "\n")
            text_widget.write(f'If this error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            torch.cuda.empty_cache()
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'Missing file error raised.\n' + 
                            "\n" + f'Please address the error and try again.' + "\n" +
                            f'If this error persists, please contact the developers.\n\n' +
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if ffmp_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The input file type is not supported or FFmpeg is missing.\n')
            text_widget.write(f'Please select a file type supported by FFmpeg and try again.\n\n')
            text_widget.write(f'If FFmpeg is missing or not installed, you will only be able to process \".wav\" files \nuntil it is available on this system.\n\n')
            text_widget.write(f'See the \"More Info\" tab in the Help Guide.\n\n')
            text_widget.write(f'If this error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            torch.cuda.empty_cache()
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'The input file type is not supported or FFmpeg is missing.\nPlease select a file type supported by FFmpeg and try again.\n\n' + 
                            f'If FFmpeg is missing or not installed, you will only be able to process \".wav\" files until it is available on this system.\n\n' + 
                            f'See the \"More Info\" tab in the Help Guide.\n\n' + 
                            f'If this error persists, please contact the developers.\n\n' +
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if onnxmissing in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application could not detect this MDX-Net model on your system.\n')
            text_widget.write(f'Please make sure all the models are present in the correct directory.\n')
            text_widget.write(f'If the error persists, please reinstall application or contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'The application could not detect this MDX-Net model on your system.\n' + 
                            f'Please make sure all the models are present in the correct directory.\n' + 
                            f'If the error persists, please reinstall application or contact the developers.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if onnxmemerror in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough GPU memory to use this model.\n')
            text_widget.write(f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n')
            text_widget.write(f'If the error persists, your GPU might not be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'The application was unable to allocate enough GPU memory to use this model.\n' + 
                            f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n' + 
                            f'If the error persists, your GPU might not be supported.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if onnxmemerror2 in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough GPU memory to use this model.\n')
            text_widget.write(f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n')
            text_widget.write(f'If the error persists, your GPU might not be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'The application was unable to allocate enough GPU memory to use this model.\n' + 
                            f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n' + 
                            f'If the error persists, your GPU might not be supported.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if sf_write_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Could not write audio file.\n')
            text_widget.write(f'This could be due to low storage on target device or a system permissions issue.\n')
            text_widget.write(f"\nFor raw error details, go to the Error Log tab in the Help Guide.\n")
            text_widget.write(f'\nIf the error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'Could not write audio file.\n' + 
                            f'This could be due to low storage on target device or a system permissions issue.\n' + 
                            f'If the error persists, please contact the developers.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if systemmemerr in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough system memory to use this \nmodel.\n\n')
            text_widget.write(f'Please do the following:\n\n1. Restart this application.\n2. Ensure any CPU intensive applications are closed.\n3. Then try again.\n\n')
            text_widget.write(f'Please Note: Intel Pentium and Intel Celeron processors do not work well with \nthis application.\n\n')
            text_widget.write(f'If the error persists, the system may not have enough RAM, or your CPU might \nnot be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'The application was unable to allocate enough system memory to use this model.\n' + 
                            f'Please do the following:\n\n1. Restart this application.\n2. Ensure any CPU intensive applications are closed.\n3. Then try again.\n\n' + 
                            f'Please Note: Intel Pentium and Intel Celeron processors do not work well with this application.\n\n' +
                            f'If the error persists, the system may not have enough RAM, or your CPU might \nnot be supported.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if model_adv_set_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The current ONNX model settings are not compatible with the selected \nmodel.\n\n')
            text_widget.write(f'Please re-configure the advanced ONNX model settings accordingly and try \nagain.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Demucs v3\n\n' +
                            f'The current ONNX model settings are not compatible with the selected model.\n\n' + 
                            f'Please re-configure the advanced ONNX model settings accordingly and try again.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        
        print(traceback_text)
        print(type(e).__name__, e)
        print(message)
        
        try:
            with open('errorlog.txt', 'w') as f:
                f.write(f'Last Error Received:\n\n' +
                        f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                        f'Process Method: Demucs v3\n\n' +
                        f'If this error persists, please contact the developers with the error details.\n\n' +
                        message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
        except:
            tk.messagebox.showerror(master=window,
                            title='Error Details',
                            message=message)
        progress_var.set(0)
        text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
        text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
        text_widget.write(f'\nError Received:\n')
        text_widget.write("\nFor raw error details, go to the Error Log tab in the Help Guide.\n")
        text_widget.write("\n" + f'Please address the error and try again.' + "\n")
        text_widget.write(f'If this error persists, please contact the developers with the error details.\n\n')
        text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
        try:
            torch.cuda.empty_cache()
        except:
            pass
        button_widget.configure(state=tk.NORMAL)  # Enable Button
        return
    
    progress_var.set(0)

    text_widget.write(f'\nConversion(s) Completed!\n')
        
    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
    torch.cuda.empty_cache() 
    button_widget.configure(state=tk.NORMAL)  # Enable Button
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Successfully completed music demixing.");print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

