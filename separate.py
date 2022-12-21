from __future__ import annotations
from typing import TYPE_CHECKING
from demucs.apply import apply_model, demucs_segments
from demucs.hdemucs import HDemucs
from demucs.model_v2 import auto_load_demucs_model_v2
from demucs.pretrained import get_model as _gm
from demucs.utils import apply_model_v1
from demucs.utils import apply_model_v2
from lib_v5 import spec_utils
from lib_v5.vr_network import nets
from lib_v5.vr_network import nets_new
#from lib_v5.vr_network.model_param_init import ModelParameters
from pathlib import Path
from gui_data.constants import *
import gzip
import librosa
import math
import numpy as np
import onnxruntime as ort
import os
import torch
import warnings
import pydub
import soundfile as sf

if TYPE_CHECKING:
    from UVR import ModelData

warnings.filterwarnings("ignore")
cpu = torch.device('cpu')

class SeperateAttributes:
    def __init__(self, model_data: ModelData, process_data: dict, main_model_primary_stem_4_stem=None, main_process_method=None):
        
        self.list_all_models: list
        self.process_data = process_data
        self.progress_value = 0
        self.set_progress_bar = process_data['set_progress_bar']
        self.write_to_console = process_data['write_to_console']
        self.audio_file = process_data['audio_file']
        self.audio_file_base = process_data['audio_file_base']
        self.export_path = process_data['export_path']
        self.cached_source_callback = process_data['cached_source_callback']
        self.cached_model_source_holder = process_data['cached_model_source_holder']
        self.is_4_stem_ensemble = process_data['is_4_stem_ensemble']
        self.list_all_models = process_data['list_all_models']
        self.process_iteration = process_data['process_iteration']
        self.model_samplerate = model_data.model_samplerate
        self.is_pre_proc_model = model_data.is_pre_proc_model
        self.is_secondary_model_activated = model_data.is_secondary_model_activated if not self.is_pre_proc_model else False
        self.is_secondary_model = model_data.is_secondary_model if not self.is_pre_proc_model else True
        self.process_method = model_data.process_method
        self.model_path = model_data.model_path
        self.model_name = model_data.model_name
        self.model_basename = model_data.model_basename
        self.wav_type_set = model_data.wav_type_set
        self.mp3_bit_set = model_data.mp3_bit_set
        self.save_format = model_data.save_format
        self.is_gpu_conversion = model_data.is_gpu_conversion
        self.is_normalization = model_data.is_normalization
        self.is_primary_stem_only = model_data.is_primary_stem_only if not self.is_secondary_model else model_data.is_primary_model_primary_stem_only
        self.is_secondary_stem_only = model_data.is_secondary_stem_only if not self.is_secondary_model else model_data.is_primary_model_secondary_stem_only      
        self.is_ensemble_mode = model_data.is_ensemble_mode
        self.secondary_model = model_data.secondary_model #VERIFY WHERE
        self.primary_model_primary_stem = model_data.primary_model_primary_stem
        self.primary_stem = model_data.primary_stem #-
        self.secondary_stem = model_data.secondary_stem #-
        self.is_invert_spec = model_data.is_invert_spec #
        self.secondary_model_scale = model_data.secondary_model_scale #
        self.is_demucs_pre_proc_model_inst_mix = model_data.is_demucs_pre_proc_model_inst_mix #
        #############################
        self.primary_source_map = {}
        self.secondary_source_map = {}
        self.primary_source = None
        self.secondary_source = None
        self.secondary_source_primary = None
        self.secondary_source_secondary = None

        if not model_data.process_method == DEMUCS_ARCH_TYPE:
            if process_data['is_ensemble_master'] and not self.is_4_stem_ensemble:
                if not model_data.ensemble_primary_stem == self.primary_stem:
                    self.is_primary_stem_only, self.is_secondary_stem_only = self.is_secondary_stem_only, self.is_primary_stem_only
            
            if self.is_secondary_model and not process_data['is_ensemble_master']:
                if not self.primary_model_primary_stem == self.primary_stem and not main_model_primary_stem_4_stem:
                    self.is_primary_stem_only, self.is_secondary_stem_only = self.is_secondary_stem_only, self.is_primary_stem_only
                    
            if main_model_primary_stem_4_stem:
                self.is_primary_stem_only = True if main_model_primary_stem_4_stem == self.primary_stem else False
                self.is_secondary_stem_only = True if not main_model_primary_stem_4_stem == self.primary_stem else False

            if self.is_pre_proc_model:
                self.is_primary_stem_only = True if self.primary_stem == INST_STEM else False
                self.is_secondary_stem_only = True if self.secondary_stem == INST_STEM else False

        if model_data.process_method == MDX_ARCH_TYPE:
            self.primary_model_name, self.primary_sources = self.cached_source_callback(MDX_ARCH_TYPE, model_name=self.model_basename)
            self.is_denoise = model_data.is_denoise
            self.compensate = model_data.compensate
            self.dim_f, self.dim_t = model_data.mdx_dim_f_set, 2**model_data.mdx_dim_t_set
            self.n_fft = model_data.mdx_n_fft_scale_set
            self.chunks = model_data.chunks
            self.margin = model_data.margin
            self.hop = 1024
            self.n_bins = self.n_fft//2+1
            self.chunk_size = self.hop * (self.dim_t-1)
            self.window = torch.hann_window(window_length=self.n_fft, periodic=False).to(cpu)
            self.dim_c = 4
            out_c = self.dim_c
            self.freq_pad = torch.zeros([1, out_c, self.n_bins-self.dim_f, self.dim_t]).to(cpu)
        
        if model_data.process_method == DEMUCS_ARCH_TYPE:
            self.demucs_stems = model_data.demucs_stems if not main_process_method in [MDX_ARCH_TYPE, VR_ARCH_TYPE] else None
            self.secondary_model_4_stem = model_data.secondary_model_4_stem
            self.secondary_model_4_stem_scale = model_data.secondary_model_4_stem_scale
            self.primary_stem = model_data.ensemble_primary_stem if process_data['is_ensemble_master'] else model_data.primary_stem
            self.secondary_stem = model_data.ensemble_secondary_stem if process_data['is_ensemble_master'] else model_data.secondary_stem
            self.is_chunk_demucs = model_data.is_chunk_demucs
            self.segment = model_data.segment
            self.demucs_version = model_data.demucs_version
            self.demucs_source_list = model_data.demucs_source_list
            self.demucs_source_map = model_data.demucs_source_map
            self.is_demucs_combine_stems = model_data.is_demucs_combine_stems
            self.demucs_stem_count = model_data.demucs_stem_count
            self.pre_proc_model = model_data.pre_proc_model
            
            if self.is_secondary_model and not process_data['is_ensemble_master']:
                if not self.demucs_stem_count == 2 and model_data.primary_model_primary_stem == INST_STEM:
                    self.primary_stem = VOCAL_STEM
                    self.secondary_stem = INST_STEM
                else:
                    self.primary_stem = model_data.primary_model_primary_stem
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
            
            if self.is_chunk_demucs:
                self.chunks_demucs = model_data.chunks_demucs
                self.margin_demucs = model_data.margin_demucs
            else:
                self.chunks_demucs = 0
                self.margin_demucs = 44100
                
            self.shifts = model_data.shifts
            self.is_split_mode = model_data.is_split_mode if not self.demucs_version == DEMUCS_V4 else True
            self.overlap = model_data.overlap
            self.primary_model_name, self.primary_sources = self.cached_source_callback(DEMUCS_ARCH_TYPE, model_name=self.model_basename)

        if model_data.process_method == VR_ARCH_TYPE:
            self.primary_model_name, self.primary_sources = self.cached_source_callback(VR_ARCH_TYPE, model_name=self.model_basename)
            self.mp = model_data.vr_model_param
            self.high_end_process = model_data.is_high_end_process
            self.is_tta = model_data.is_tta
            self.is_post_process = model_data.is_post_process
            self.is_gpu_conversion = model_data.is_gpu_conversion
            self.batch_size = model_data.batch_size
            self.crop_size = model_data.crop_size
            self.window_size = model_data.window_size
            self.input_high_end_h = None
            self.post_process_threshold = model_data.post_process_threshold
            self.aggressiveness = {'value': model_data.aggression_setting, 
                                   'split_bin': self.mp.param['band'][1]['crop_stop'], 
                                   'aggr_correction': self.mp.param.get('aggr_correction')}

    def start_inference(self):
        
        if self.is_secondary_model and not self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_SEC(self.process_method, self.model_basename))
        
        if self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_PRE(self.process_method, self.model_basename))
        
    def running_inference(self, is_no_write=False):
        
        self.write_to_console(DONE, base_text='') if not is_no_write else None
        self.set_progress_bar(0.05) if not is_no_write else None
        
        if self.is_secondary_model and not self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_1_SEC)
        elif self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_1_PRE)
        else:
            self.write_to_console(INFERENCE_STEP_1)
        
    def load_cached_sources(self, is_4_stem_demucs=False):
        
        if self.is_secondary_model and not self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_SEC_CACHED_MODOEL(self.process_method, self.model_basename))
        elif self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_PRE_CACHED_MODOEL(self.process_method, self.model_basename))
        else:
            self.write_to_console(INFERENCE_STEP_2_PRIMARY_CACHED)

        if not is_4_stem_demucs:
            primary_stem, secondary_stem = gather_sources(self.primary_stem, self.secondary_stem, self.primary_sources)
            
            return primary_stem, secondary_stem
            
    def cache_source(self, secondary_sources):
        
        model_occurrences = self.list_all_models.count(self.model_basename)
        
        if not model_occurrences <= 1:
            if self.process_method == MDX_ARCH_TYPE:
                self.cached_model_source_holder(MDX_ARCH_TYPE, secondary_sources, self.model_basename)
                
            if self.process_method == VR_ARCH_TYPE:
                self.cached_model_source_holder(VR_ARCH_TYPE, secondary_sources, self.model_basename)

            if self.process_method == DEMUCS_ARCH_TYPE:
                self.cached_model_source_holder(DEMUCS_ARCH_TYPE, secondary_sources, self.model_basename)
                
            # if isinstance(secondary_sources, np.ndarray):
            #     print('\n==================================\n', secondary_sources, f"\n\nMemory size of source for model {self.model_basename}: ", secondary_sources.size * secondary_sources.itemsize, 'BYTES', '\n==================================\n')
                
            # if type(secondary_sources) is dict:
            #     print('\n==================================\n', secondary_sources, f"\n\nMemory size of source for model {self.model_basename}: ", sum(v.size * v.itemsize for v in secondary_sources.values()), 'BYTES', '\n==================================\n')

    def write_audio(self, stem_path, stem_source, samplerate, secondary_model_source=None, model_scale=None):
                
        if not self.is_secondary_model:
            if self.is_secondary_model_activated:
                if isinstance(secondary_model_source, np.ndarray):
                    secondary_model_scale = model_scale if model_scale else self.secondary_model_scale
                    stem_source = spec_utils.average_dual_sources(stem_source, secondary_model_source, secondary_model_scale)
            
            sf.write(stem_path, stem_source, samplerate, subtype=self.wav_type_set)
            save_format(stem_path, self.save_format, self.mp3_bit_set) if not self.is_ensemble_mode else None
            
            self.write_to_console(DONE, base_text='')
            self.set_progress_bar(0.95)

class SeperateMDX(SeperateAttributes):        

    def seperate(self):
        
        samplerate = 44100
                
        if self.primary_model_name == self.model_basename and self.primary_sources:
            self.primary_source, self.secondary_source = self.load_cached_sources()
        else:
            self.start_inference()
            if self.is_gpu_conversion >= 0:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
                run_type = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            else:
                self.device = torch.device('cpu')
                run_type = ['CPUExecutionProvider']

            self.onnx_model = ort.InferenceSession(self.model_path, providers=run_type)

            self.running_inference()
            mdx_net_cut = True if self.primary_stem in MDX_NET_FREQ_CUT else False
            mix, raw_mix, samplerate = prepare_mix(self.audio_file, self.chunks, self.margin, mdx_net_cut=mdx_net_cut)
            
            source = self.demix_base(mix)
            self.write_to_console(DONE, base_text='')            

        if self.is_secondary_model_activated:
            if self.secondary_model:
                self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method)
        
        if not self.is_secondary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.primary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = spec_utils.normalize(source[0], self.is_normalization).T
            self.primary_source_map = {self.primary_stem: self.primary_source}
            self.write_audio(primary_stem_path, self.primary_source, samplerate, self.secondary_source_primary)

        if not self.is_primary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.secondary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                raw_mix = self.demix_base(raw_mix, is_match_mix=True)[0] if mdx_net_cut else raw_mix
                self.secondary_source, raw_mix = spec_utils.normalize_two_stem(source[0]*self.compensate, raw_mix, self.is_normalization)
            
                if self.is_invert_spec:
                    self.secondary_source = spec_utils.invert_stem(raw_mix, self.secondary_source)
                else:
                    self.secondary_source = (-self.secondary_source.T+raw_mix.T)

            self.secondary_source_map = {self.secondary_stem: self.secondary_source}
            self.write_audio(secondary_stem_path, self.secondary_source, samplerate, self.secondary_source_secondary)

        torch.cuda.empty_cache()

        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}

        self.cache_source(secondary_sources)

        if self.is_secondary_model:
            return secondary_sources

    def demix_base(self, mix, is_match_mix=False):
        chunked_sources = []

        for slice in mix:
            self.progress_value += 1
            self.set_progress_bar(0.1, (0.8/len(mix)*self.progress_value)) if not is_match_mix else None
            cmix = mix[slice]
            sources = []
            mix_waves = []
            n_sample = cmix.shape[1]
            trim = self.n_fft//2
            gen_size = self.chunk_size-2*trim
            pad = gen_size - n_sample%gen_size
            mix_p = np.concatenate((np.zeros((2,trim)), cmix, np.zeros((2,pad)), np.zeros((2,trim))), 1)
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i+self.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(cpu)
            with torch.no_grad():
                _ort = self.onnx_model if not is_match_mix else None
                adjust = 1
                spek = self.stft(mix_waves)*adjust
                
                if not is_match_mix:
                    if self.is_denoise:
                        spec_pred = -_ort.run(None, {'input': -spek.cpu().numpy()})[0]*0.5+_ort.run(None, {'input': spek.cpu().numpy()})[0]*0.5
                    else:
                        spec_pred = _ort.run(None, {'input': spek.cpu().numpy()})[0]
                else:
                    spec_pred = spek.cpu().numpy()

                tar_waves = self.istft(torch.tensor(spec_pred))#.cpu()
                tar_signal = tar_waves[:,:,trim:-trim].transpose(0,1).reshape(2, -1).numpy()[:, :-pad]
                start = 0 if slice == 0 else self.margin
                end = None if slice == list(mix.keys())[::-1][0] else -self.margin
                if self.margin == 0:
                    end = None
                sources.append(tar_signal[:,start:end]*(1/adjust))
            chunked_sources.append(sources)
        sources = np.concatenate(chunked_sources, axis=-1)

        if not is_match_mix:
            del self.onnx_model

        return sources
    
    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,self.dim_c,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        c = 2
        x = x.reshape([-1,c,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,c,self.chunk_size])

class SeperateDemucs(SeperateAttributes):        

    def seperate(self):

        samplerate = 44100
        source = None
        model_scale = None
        stem_source = None
        stem_source_secondary = None
        inst_mix = None
        inst_raw_mix = None
        raw_mix = None
        inst_source = None
        is_no_write = False
        is_no_piano_guitar = False

        if self.primary_model_name == self.model_basename and type(self.primary_sources) is dict and not self.pre_proc_model:
            self.primary_source, self.secondary_source = self.load_cached_sources()
        elif self.primary_model_name == self.model_basename and isinstance(self.primary_sources, np.ndarray) and not self.pre_proc_model:
            source = self.primary_sources
            self.load_cached_sources(is_4_stem_demucs=True)
        else:
            self.start_inference()
            if self.is_gpu_conversion >= 0:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
            else:
                self.device = torch.device('cpu')
            
            if self.demucs_version == DEMUCS_V1:
                if str(self.model_path).endswith(".gz"):
                    self.model_path = gzip.open(self.model_path, "rb")
                klass, args, kwargs, state = torch.load(self.model_path)
                self.demucs = klass(*args, **kwargs)
                self.demucs.to(self.device) 
                self.demucs.load_state_dict(state)
            elif self.demucs_version == DEMUCS_V2:
                self.demucs = auto_load_demucs_model_v2(self.demucs_source_list, self.model_path)
                self.demucs.to(self.device) 
                self.demucs.load_state_dict(torch.load(self.model_path))
                self.demucs.eval()
            else:  
                self.demucs = HDemucs(sources=self.demucs_source_list)
                self.demucs = _gm(name=os.path.splitext(os.path.basename(self.model_path))[0], 
                                  repo=Path(os.path.dirname(self.model_path)))
                self.demucs = demucs_segments(self.segment, self.demucs)
                self.demucs.to(self.device)
                self.demucs.eval()

            if self.pre_proc_model:
                if self.primary_stem not in [VOCAL_STEM, INST_STEM]:
                    is_no_write = True
                    self.write_to_console(DONE, base_text='')
                    mix_no_voc = process_secondary_model(self.pre_proc_model, self.process_data, is_pre_proc_model=True)
                    inst_mix, inst_raw_mix, inst_samplerate = prepare_mix(mix_no_voc[INST_STEM], self.chunks_demucs, self.margin_demucs)
                    self.process_iteration()
                    self.running_inference(is_no_write=is_no_write)
                    inst_source = self.demix_demucs(inst_mix)
                    self.process_iteration()

            self.running_inference(is_no_write=is_no_write) if not self.pre_proc_model else None
            mix, raw_mix, samplerate = prepare_mix(self.audio_file, self.chunks_demucs, self.margin_demucs)
            
            if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, np.ndarray) and self.pre_proc_model:
                source = self.primary_sources
            else:
                source = self.demix_demucs(mix)
            
            self.write_to_console(DONE, base_text='')
            
            del self.demucs

        if isinstance(inst_source, np.ndarray):
            source_reshape = spec_utils.reshape_sources(inst_source[self.demucs_source_map[VOCAL_STEM]], source[self.demucs_source_map[VOCAL_STEM]])
            inst_source[self.demucs_source_map[VOCAL_STEM]] = source_reshape
            source = inst_source

        if isinstance(source, np.ndarray):
            if len(source) == 2:
                self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER
            else:
                self.demucs_source_map = DEMUCS_6_SOURCE_MAPPER if len(source) == 6 else DEMUCS_4_SOURCE_MAPPER
                if len(source) == 6 and self.process_data['is_ensemble_master'] or len(source) == 6 and self.is_secondary_model:
                    is_no_piano_guitar = True
                    six_stem_other_source = list(source)
                    six_stem_other_source = [i for n, i in enumerate(source) if n in [self.demucs_source_map[OTHER_STEM], self.demucs_source_map[GUITAR_STEM], self.demucs_source_map[PIANO_STEM]]]
                    other_source = np.zeros_like(six_stem_other_source[0])
                    for i in six_stem_other_source:
                        other_source += i
                    source_reshape = spec_utils.reshape_sources(source[self.demucs_source_map[OTHER_STEM]], other_source)
                    source[self.demucs_source_map[OTHER_STEM]] = source_reshape

        if (self.demucs_stems == ALL_STEMS and not self.process_data['is_ensemble_master']) or self.is_4_stem_ensemble:
            self.cache_source(source)
            
            for stem_name, stem_value in self.demucs_source_map.items():

                if self.is_secondary_model_activated and not self.is_secondary_model and not stem_value >= 4:
                    if self.secondary_model_4_stem[stem_value]:
                        model_scale = self.secondary_model_4_stem_scale[stem_value]
                        stem_source_secondary = process_secondary_model(self.secondary_model_4_stem[stem_value], self.process_data, main_model_primary_stem_4_stem=stem_name, is_4_stem_demucs=True)
                        if isinstance(stem_source_secondary, np.ndarray):
                            stem_source_secondary = stem_source_secondary[1 if self.secondary_model_4_stem[stem_value].demucs_stem_count == 2 else stem_value]
                            stem_source_secondary = spec_utils.normalize(stem_source_secondary, self.is_normalization).T
                        elif type(stem_source_secondary) is dict:
                            stem_source_secondary = stem_source_secondary[stem_name]
                            
                stem_source_secondary = None if stem_value >= 4 else stem_source_secondary
                self.write_to_console(f'{SAVING_STEM[0]}{stem_name}{SAVING_STEM[1]}') if not self.is_secondary_model else None
                stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({stem_name}).wav')
                stem_source = spec_utils.normalize(source[stem_value], self.is_normalization).T
                self.write_audio(stem_path, stem_source, samplerate, secondary_model_source=stem_source_secondary, model_scale=model_scale)

            if self.is_secondary_model:    
                return source
        else:
            if self.is_secondary_model_activated:
                if self.secondary_model:
                    self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method)

            if not self.is_secondary_stem_only:
                self.write_to_console(f'{SAVING_STEM[0]}{self.primary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
                if not isinstance(self.primary_source, np.ndarray):
                    self.primary_source = spec_utils.normalize(source[self.demucs_source_map[self.primary_stem]], self.is_normalization).T
                self.primary_source_map = {self.primary_stem: self.primary_source}
                self.write_audio(primary_stem_path, self.primary_source, samplerate, self.secondary_source_primary)

            if not self.is_primary_stem_only:
                def secondary_save(sec_stem_name, source, raw_mixture=None, is_inst_mixture=False):
                    secondary_source = self.secondary_source if not is_inst_mixture else None
                    self.write_to_console(f'{SAVING_STEM[0]}{sec_stem_name}{SAVING_STEM[1]}') if not self.is_secondary_model else None
                    secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({sec_stem_name}).wav')
                    secondary_source_secondary = None
                    
                    if not isinstance(secondary_source, np.ndarray):
                        if self.is_demucs_combine_stems:
                            source = list(source)
                            if is_inst_mixture:
                                source = [i for n, i in enumerate(source) if not n in [self.demucs_source_map[self.primary_stem], self.demucs_source_map[VOCAL_STEM]]]
                            else:
                                source.pop(self.demucs_source_map[self.primary_stem])
                                
                            source = source[:len(source) - 2] if is_no_piano_guitar else source
                            secondary_source = np.zeros_like(source[0])
                            for i in source:
                                secondary_source += i
                            secondary_source = spec_utils.normalize(secondary_source, self.is_normalization).T
                        else:
                            if not isinstance(raw_mixture, np.ndarray):
                                raw_mixture = prepare_mix(self.audio_file, self.chunks_demucs, self.margin_demucs, is_missing_mix=True)
       
                            secondary_source, raw_mixture = spec_utils.normalize_two_stem(source[self.demucs_source_map[self.primary_stem]], raw_mixture, self.is_normalization)
                            
                            if self.is_invert_spec:
                                secondary_source = spec_utils.invert_stem(raw_mixture, secondary_source)
                            else:
                                raw_mixture = spec_utils.reshape_sources(secondary_source, raw_mixture)
                                secondary_source = (-secondary_source.T+raw_mixture.T)
                            
                    if not is_inst_mixture:
                        self.secondary_source = secondary_source
                        secondary_source_secondary = self.secondary_source_secondary
                        self.secondary_source_map = {self.secondary_stem: self.secondary_source}

                    self.write_audio(secondary_stem_path, secondary_source, samplerate, secondary_source_secondary)

                secondary_save(self.secondary_stem, source, raw_mixture=raw_mix)
                
                if self.is_demucs_pre_proc_model_inst_mix and self.pre_proc_model and not self.is_4_stem_ensemble:
                    secondary_save(f"{self.secondary_stem} {INST_STEM}", source, raw_mixture=inst_raw_mix, is_inst_mixture=True)
                
            torch.cuda.empty_cache()
                
            secondary_sources = {**self.primary_source_map, **self.secondary_source_map}

            self.cache_source(secondary_sources)
            
            if self.is_secondary_model:    
                return secondary_sources
    
    def demix_demucs(self, mix):
        processed = {}

        set_progress_bar = None if self.is_chunk_demucs else self.set_progress_bar

        for nmix in mix:
            self.progress_value += 1
            self.set_progress_bar(0.1, (0.8/len(mix)*self.progress_value)) if self.is_chunk_demucs else None
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            mix_infer = cmix 
            
            with torch.no_grad():
                if self.demucs_version == DEMUCS_V1:
                    sources = apply_model_v1(self.demucs, 
                                                mix_infer.to(self.device), 
                                                self.shifts, 
                                                self.is_split_mode,
                                                set_progress_bar=set_progress_bar)
                elif self.demucs_version == DEMUCS_V2:
                    sources = apply_model_v2(self.demucs, 
                                                mix_infer.to(self.device), 
                                                self.shifts,
                                                self.is_split_mode,
                                                self.overlap,
                                                set_progress_bar=set_progress_bar)
                else:
                    sources = apply_model(self.demucs, 
                                            mix_infer[None], 
                                            self.shifts,
                                            self.is_split_mode,
                                            self.overlap,
                                            static_shifts=1 if self.shifts == 0 else self.shifts,
                                            set_progress_bar=set_progress_bar,
                                            device=self.device)[0]
            
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]
            start = 0 if nmix == 0 else self.margin_demucs
            end = None if nmix == list(mix.keys())[::-1][0] else -self.margin_demucs
            if self.margin_demucs == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()
            sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)
                        
        return sources

class SeperateVR(SeperateAttributes):        

    def seperate(self):

        if self.primary_model_name == self.model_basename and self.primary_sources:
            self.primary_source, self.secondary_source = self.load_cached_sources()
        else:
            self.start_inference()
            # if self.is_gpu_conversion >= 0:
            #     #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
            # else:
            #     device = torch.device('cpu')
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #MACOS_COMMENT
            
            nn_arch_sizes = [
                31191, # default
                33966, 56817, 218409, 123821, 123812, 129605, 537238, 537227]
            vr_5_1_models = [56817, 218409]
            
            model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
            nn_architecture = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

            #print('ARC SIZE: ', nn_architecture)

            if nn_architecture in vr_5_1_models:
                model = nets_new.CascadedNet(self.mp.param['bins'] * 2, nn_architecture)
                inference = self.inference_vr_new
            else:
                model = nets.determine_model_capacity(self.mp.param['bins'] * 2, nn_architecture)
                inference = self.inference_vr

            model.load_state_dict(torch.load(self.model_path, map_location=device)) 
            model.to(device) 
            
            self.running_inference()
            
            y_spec, v_spec = inference(self.loading_mix(), device, model, self.aggressiveness)
            self.write_to_console(DONE, base_text='')
            
            del model

        if self.is_secondary_model_activated:
            if self.secondary_model:
                self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method)

        if not self.is_secondary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.primary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = spec_utils.normalize(self.spec_to_wav(y_spec), self.is_normalization).T
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                
            self.primary_source_map = {self.primary_stem: self.primary_source}
            
            self.write_audio(primary_stem_path, self.primary_source, 44100, self.secondary_source_primary)

        if not self.is_primary_stem_only:
            self.write_to_console(f'{SAVING_STEM[0]}{self.secondary_stem}{SAVING_STEM[1]}') if not self.is_secondary_model else None
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec)
                self.secondary_source = spec_utils.normalize(self.spec_to_wav(v_spec), self.is_normalization).T
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.secondary_source_map = {self.secondary_stem: self.secondary_source}
            
            self.write_audio(secondary_stem_path, self.secondary_source, 44100, self.secondary_source_secondary)

        torch.cuda.empty_cache()
        
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        self.cache_source(secondary_sources)

        if self.is_secondary_model:
            return secondary_sources
            
    def loading_mix(self):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(self.mp.param['band'])
        
        for d in range(bands_n, 0, -1):        
            bp = self.mp.param['band'][d]
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(
                    self.audio_file, bp['sr'], False, dtype=np.float32, res_type='polyphase') #MACOS_COMMENT
                    
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type='polyphase') #MACOS_COMMENT
                
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], self.mp.param['mid_side'], 
                                                            self.mp.param['mid_side_b2'], self.mp.param['reverse'])
            
            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        
        del X_wave, X_spec_s

        return X_spec

    def inference_vr(self, X_spec, device, model, aggressiveness):
        
        def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness):
            model.eval()
            
            total_iterations = sum([n_window]) if not self.is_tta else sum([n_window])*2
            
            with torch.no_grad():
                preds = []

                for i in range(n_window):
                    self.progress_value +=1
                    self.set_progress_bar(0.1, 0.8/total_iterations*self.progress_value)
                    start = i * roi_size
                    X_mag_window = X_mag_pad[None, :, :, start:start + self.window_size]
                    X_mag_window = torch.from_numpy(X_mag_window).to(device)
                    pred = model.predict(X_mag_window, aggressiveness)
                    pred = pred.detach().cpu().numpy()
                    preds.append(pred[0])
                    
                pred = np.concatenate(preds, axis=2)
            return pred

        X_mag, X_phase = spec_utils.preprocess(X_spec)
        coef = X_mag.max()
        X_mag_pre = X_mag / coef
        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, model.offset)
        n_window = int(np.ceil(n_frame / roi_size))
        X_mag_pad = np.pad(
            X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        pred = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness)
        pred = pred[:, :, :n_frame]
        
        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1
            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            pred_tta = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness)
            pred_tta = pred_tta[:, :, roi_size // 2:]
            pred_tta = pred_tta[:, :, :n_frame]
            pred, X_mag, X_phase = (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)
        else:
            pred, X_mag, X_phase = pred * coef, X_mag, np.exp(1.j * X_phase)
                   
        if self.is_post_process:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv, thres=self.post_process_threshold)
            
        y_spec = pred * X_phase
        v_spec = X_spec - y_spec
        
        return y_spec, v_spec

    def inference_vr_new(self, X_spec, device, model, aggressiveness):
        
        def _execute(X_mag_pad, roi_size):
            
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * model.offset) // roi_size
            total_iterations = patches//self.batch_size if not self.is_tta else (patches//self.batch_size)*2
            
            for i in range(patches):
                start = i * roi_size
                X_mag_crop = X_mag_pad[:, :, start:start + self.crop_size]
                X_dataset.append(X_mag_crop)

            X_dataset = np.asarray(X_dataset)
            model.eval()
            
            with torch.no_grad():
                mask = []
                # To reduce the overhead, dataloader is not used.
                for i in range(0, patches, self.batch_size):
                    self.progress_value += 1
                    if self.progress_value >= total_iterations:
                        self.progress_value = total_iterations
                    self.set_progress_bar(0.1, 0.8/total_iterations*self.progress_value)
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = model.predict_mask(X_batch)
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                    
                mask = np.concatenate(mask, axis=2)

            return mask
        
        def postprocess(mask, X_mag, X_phase, aggressiveness):
            
            if self.primary_stem == VOCAL_STEM:
                mask = (1.0 - spec_utils.adjust_aggr(mask, True, aggressiveness))
            else:   
                mask = spec_utils.adjust_aggr(mask, False, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.crop_size, model.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        
        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase, aggressiveness)
        
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        
        if self.high_end_process.startswith('mirroring'):        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp)
            
        return wav
   
def process_secondary_model(secondary_model: ModelData, process_data, main_model_primary_stem_4_stem=None, is_4_stem_demucs=False, main_process_method=None, is_pre_proc_model=False):
        
    if not is_pre_proc_model:
        process_iteration = process_data['process_iteration']
        process_iteration()
    
    if secondary_model.process_method == VR_ARCH_TYPE:
        seperator = SeperateVR(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method)
    if secondary_model.process_method == MDX_ARCH_TYPE:
        seperator = SeperateMDX(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method)
    if secondary_model.process_method == DEMUCS_ARCH_TYPE:
        seperator = SeperateDemucs(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method)
        
    secondary_sources = seperator.seperate()

    if type(secondary_sources) is dict and not is_4_stem_demucs and not is_pre_proc_model:
        return gather_sources(secondary_model.primary_model_primary_stem, STEM_PAIR_MAPPER[secondary_model.primary_model_primary_stem], secondary_sources)
    else:
        return secondary_sources
    
def gather_sources(primary_stem_name, secondary_stem_name, secondary_sources: dict):
    
    source_primary = False
    source_secondary = False

    for key, value in secondary_sources.items():
        if key in primary_stem_name:
            source_primary = value
        if key in secondary_stem_name:
            source_secondary = value

    return source_primary, source_secondary
        
def prepare_mix(mix, chunk_set, margin_set, mdx_net_cut=False, is_missing_mix=False):

    samplerate = 44100

    if not isinstance(mix, np.ndarray):
        mix, samplerate = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    def get_segmented_mix(chunk_set=chunk_set):
        segmented_mix = {}
        
        samples = mix.shape[-1]
        margin = margin_set
        chunk_size = chunk_set*44100
        assert not margin == 0, 'margin cannot be zero!'
        if margin > chunk_size:
            margin = chunk_size
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
            
        return segmented_mix

    if is_missing_mix:
        return mix
    else:
        segmented_mix = get_segmented_mix()
        raw_mix = get_segmented_mix(chunk_set=0) if mdx_net_cut else mix
        return segmented_mix, raw_mix, samplerate

def save_format(audio_path, save_format, mp3_bit_set):
    
    if not save_format == WAV:
        
        FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg')
        
        pydub.AudioSegment.converter = FFMPEG_PATH
        musfile = pydub.AudioSegment.from_wav(audio_path)

        if save_format == FLAC:
            audio_path_flac = audio_path.replace(".wav", ".flac")
            musfile.export(audio_path_flac, format="flac")  
        
        if save_format == MP3:
            audio_path_mp3 = audio_path.replace(".wav", ".mp3")
            musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set)
        
        try:
            os.remove(audio_path)
        except Exception as e:
            print(e)
