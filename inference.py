import argparse
import os
import importlib

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import time
import math
from tqdm import tqdm

from lib import dataset
from lib import spec_utils
from lib.model_param_init import ModelParameters


class VocalRemover(object):

    def __init__(self, model, device, window_size):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.window_size = window_size

    def _execute(self, X_mag_pad, roi_size, n_window, params):
        self.model.eval()
        with torch.no_grad():
            preds = []
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start:start + self.window_size]
                X_mag_window = torch.from_numpy(X_mag_window).to(self.device)

                pred = self.model.predict(X_mag_window, params)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)

        return pred

    def preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def inference(self, X_spec, params):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred = self._execute(X_mag_pad, roi_size, n_window, params)
        pred = pred[:, :, :n_frame]

        return pred * coef, X_mag, np.exp(1.j * X_phase)

    def inference_tta(self, X_spec, params):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred = self._execute(X_mag_pad, roi_size, n_window, params)
        pred = pred[:, :, :n_frame]

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred_tta = self._execute(X_mag_pad, roi_size, n_window, params)
        pred_tta = pred_tta[:, :, roi_size // 2:]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)
        
        
def main():
    nn_arch_sizes = [
        31191, # default
        33966, 123821, 537238 # custom
    ]

    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--nn_architecture', '-n', type=str, choices= ['auto'] + list('{}KB'.format(s) for s in nn_arch_sizes), default='auto')
    p.add_argument('--model_params', '-m', type=str, default='')
    p.add_argument('--window_size', '-w', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--deepextraction', '-D', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--is_vocal_model', '-vm', action='store_true')
    p.add_argument('--tta', '-t', action='store_true', help='Test-Time-Augmentation')
    p.add_argument('--high_end_process', '-H', type=str, choices=['mirroring', 'mirroring2', 'bypass', 'none'], default='mirroring')
    p.add_argument('--aggressiveness', '-A', type=float, default=0.07, help='The strength of the vocal isolation. From 0.0 to 1.0.')
    p.add_argument('--no_vocals', '-nv', action='store_true', help='Don\'t create Vocals stem.')
    p.add_argument('--chunks', '-c', type=int, default=1, help='Split the input file into chunks to reduce RAM consumption.')
    p.add_argument('--model_test_mode', '-mt', action='store_true', help='Include the model name in the output file name.')
    p.add_argument('--normalize', action='store_true')
    args = p.parse_args()
    
    separated_dir = 'separated'
    ensembled_dir = 'ensembled/temp'
    for file in os.scandir(ensembled_dir):
        os.remove(file.path)
    
    if 'auto' == args.nn_architecture:
        model_size = math.ceil(os.stat(args.pretrained_model).st_size / 1024)
        args.nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
        
    nets = importlib.import_module('lib.nets' + f'_{args.nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)

    mp = ModelParameters(args.model_params)       
    start_time = time.time()

    print('loading model...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_spec = {}
    input_is_mono = False
    basename = os.path.splitext(os.path.basename(args.input))[0]
    basenameb = '"{}"'.format(os.path.splitext(os.path.basename(args.input))[0])
    bands_n = len(mp.param['band'])
    
    # high-end band
    bp = mp.param['band'][bands_n]
    wave, _ = librosa.load(args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])

    if wave.ndim == 1:
        input_is_mono = True
        wave = np.asarray([wave, wave])
        
    if args.normalize:
        wave /= max(np.max(wave), abs(np.min(wave)))
    
    X_spec[bands_n] = spec_utils.wave_to_spectrogram(wave, bp['hl'], bp['n_fft'], mp, True)
    X_spec[bands_n] = spec_utils.convert_channels(X_spec[bands_n], mp, bands_n)
        
    if np.max(wave[0]) == 0.0:
        print('Empty audio file!')
        raise ValueError('Empty audio file')
        
    if args.high_end_process != 'none':
        input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
        input_high_end = X_spec[bands_n][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]
             
    # lower bands
    for d in range(bands_n - 1, 0, -1):        
        bp = mp.param['band'][d]
        
        wave = librosa.resample(wave, mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
        X_spec[d] = spec_utils.wave_to_spectrogram(wave, bp['hl'], bp['n_fft'], mp, True)
        X_spec[d] = spec_utils.convert_channels(X_spec[d], mp, d)
    
    X_spec = spec_utils.combine_spectrograms(X_spec, mp)
        
    print('done')
        
    vr = VocalRemover(model, device, args.window_size)
    
    chunk_pfx = ''
    chunk_size = X_spec.shape[2] // args.chunks
    chunks_filelist = {'vocals': {}, 'inst': {}}
    
    for chunk in range(0, args.chunks):
        chunk_margin_r = 0
    
        if chunk == 0:
            chunk_offset_m, chunk_offset, chunk_margin = 0, 0, 0
        else:
            chunk_margin = chunk_size // 100 - 1
            chunk_offset_m = chunk * chunk_size - chunk_margin - 1
            chunk_offset = chunk * chunk_size - 1
        
        if args.chunks > 1:
            chunk_pfx = f'_chunk{chunk}'
            print(f'Chunk {chunk}')
            
            if chunk < args.chunks - 1:
                chunk_margin_r = chunk_size // 100 - 1
        
        pd = {
            'aggr_value': args.aggressiveness,
            'aggr_split_bin': mp.param['band'][1]['crop_stop'],
            'aggr_correction': mp.param.get('aggr_correction'),
            'is_vocal_model': args.is_vocal_model
        }

        if args.tta:
            pred, X_mag, X_phase = vr.inference_tta(X_spec[:, :, chunk_offset_m:(chunk+1)*chunk_size+chunk_margin_r], pd)
        else:
            pred, X_mag, X_phase = vr.inference(X_spec[:, :, chunk_offset_m:(chunk+1)*chunk_size+chunk_margin_r], pd)

        if args.postprocess:
            print('post processing...', end=' ')
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
            print('done')

        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}        
        basename_enc = basename
            
        print('inverse stft of {}...'.format(stems['inst']), end=' ')
        y_spec_m = (pred * X_phase)[:, :, chunk_margin:pred.shape[2]-chunk_margin_r]
        
        if args.chunks > 1:
            import hashlib
        
            basename_enc = hashlib.sha1(basename.encode('utf-8')).hexdigest()
        
            if chunk > 0: # smoothing
                y_spec_m[:, :, 0] = 0.5 * (y_spec_m[:, :, 0] + prev_chunk_edge)
            prev_chunk_edge = y_spec_m[:, :, -1]
        
        ffmpeg_tmp_fn = '{}_{}_inst'.format(basename_enc, time.time())
       
        if args.high_end_process == 'bypass':
            wave = spec_utils.cmb_spectrogram_to_wave_ffmpeg(y_spec_m, mp, ffmpeg_tmp_fn, input_high_end_h, input_high_end)
        elif args.high_end_process.startswith('mirroring'):        
            input_high_end_ = spec_utils.mirroring(args.high_end_process, y_spec_m, input_high_end[:, :, chunk_offset:(chunk+1)*chunk_size], mp)
            
            wave = spec_utils.cmb_spectrogram_to_wave_ffmpeg(y_spec_m, mp, ffmpeg_tmp_fn, input_high_end_h, input_high_end_)       
        else:
            wave = spec_utils.cmb_spectrogram_to_wave_ffmpeg(y_spec_m, mp, ffmpeg_tmp_fn)
        
        print('done')
        
        model_name = ''
        
        if args.model_test_mode:
            model_name = '_' + os.path.splitext(os.path.basename(args.pretrained_model))[0]
            
        if input_is_mono:
            wave = wave.mean(axis=1, keepdims=True)
        
        fn = os.path.join(separated_dir, '{}{}_{}{}.wav'.format(basename_enc, model_name, stems['inst'], chunk_pfx))        
        sf.write(fn, wave, mp.param['sr'])
        chunks_filelist['inst'][chunk] = fn

        if not args.no_vocals:
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')
            
            ffmpeg_tmp_fn = '{}_{}_vocals'.format(basename_enc, time.time())
            v_spec_m = X_spec[:, :, chunk_offset:(chunk+1)*chunk_size] - y_spec_m

            if args.high_end_process.startswith('mirroring'):        
                input_high_end_ = spec_utils.mirroring(args.high_end_process, v_spec_m, input_high_end[:, :, chunk_offset:(chunk+1)*chunk_size], mp)

                wave = spec_utils.cmb_spectrogram_to_wave_ffmpeg(v_spec_m, mp, ffmpeg_tmp_fn, input_high_end_h, input_high_end_)       
            else:        
                wave = spec_utils.cmb_spectrogram_to_wave_ffmpeg(v_spec_m, mp, ffmpeg_tmp_fn)
                
            print('done')
            
            if input_is_mono:
                wave = wave.mean(axis=1, keepdims=True)
            
            fn = os.path.join(separated_dir, '{}{}_{}{}.wav'.format(basename_enc, model_name, stems['vocals'], chunk_pfx))
            sf.write(fn, wave, mp.param['sr'])
            chunks_filelist['vocals'][chunk] = fn
        
    for stem in stems:
        if len(chunks_filelist[stem]) > 0 and args.chunks > 1: 
            import subprocess
        
            fn = os.path.join(separated_dir, '{}{}_{}.wav'.format(basename_enc, model_name, stems[stem]))  
            fn2 = os.path.join(separated_dir, '{}{}_{}.wav'.format(basename, model_name, stems[stem]))  
            #os.system('sox "' + '" "'.join([f for f in chunks_filelist[stem].values()]) + f'" "{fn}"')
            subprocess.run(['sox'] + [f for f in chunks_filelist[stem].values()] + [fn])
            
            if not os.path.isfile(fn):
                print('Error: failed to create output file. Make sure that you have installed sox.')
                
            os.rename(fn, fn2)
            
            for rf in chunks_filelist[stem].values():
                os.remove(rf)

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
            
        if not args.no_vocals:
            with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
                image = spec_utils.spectrogram_to_image(v_spec_m)
                _, bin_image = cv2.imencode('.jpg', image)
                bin_image.tofile(f)

    if args.deepextraction:

        deepext = [
            {
                'algorithm':'deep',
                'model_params':'modelparams/1band_sr44100_hl512.json',
                'file1':"{}/{}{}_{}.wav".format(separated_dir, basenameb, model_name, stems['vocals'], mp.param['sr']),
                'file2':"{}/{}{}_{}.wav".format(separated_dir, basenameb, model_name, stems['inst'], mp.param['sr']),
                'output':'{}/{}{}_{}_Deep_Extraction'.format(separated_dir, basenameb, model_name, stems['inst'], mp.param['sr'])
            }
        ]

        for i,e in tqdm(enumerate(deepext), desc="Performing Deep Extraction..."):
            os.system(f"python lib/spec_utils.py -a {e['algorithm']} -m {e['model_params']} {e['file1']} {e['file2']} -o {e['output']}")
     
    for file in os.scandir(ensembled_dir):
        os.remove(file.path)
    print('Complete!')

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))


if __name__ == '__main__':
    main()

