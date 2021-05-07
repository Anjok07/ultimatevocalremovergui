import argparse
import os, glob

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import time
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

    def _execute(self, X_mag_pad, roi_size, n_window, aggressiveness):
        self.model.eval()
        with torch.no_grad():
            preds = []
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start:start + self.window_size]
                X_mag_window = torch.from_numpy(X_mag_window).to(self.device)

                pred = self.model.predict(X_mag_window, aggressiveness)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)

        return pred

    def preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def inference(self, X_spec, aggressiveness):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred = self._execute(X_mag_pad, roi_size, n_window, aggressiveness)
        pred = pred[:, :, :n_frame]

        return pred * coef, X_mag, np.exp(1.j * X_phase)

    def inference_tta(self, X_spec, aggressiveness):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred = self._execute(X_mag_pad, roi_size, n_window, aggressiveness)
        pred = pred[:, :, :n_frame]

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred_tta = self._execute(X_mag_pad, roi_size, n_window, aggressiveness)
        pred_tta = pred_tta[:, :, roi_size // 2:]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_modelA', '-Am', type=str, default='models/MGM-v5-4Band-44100-BETA1.pth') ##DON'T CHNGE ITERATION-1
    p.add_argument('--pretrained_modelB', '-B', type=str, default='models/MGM-v5-4Band-44100-BETA2.pth') ##DON'T CHNGE ITERATION-2
    p.add_argument('--pretrained_modelC', '-C', type=str, default='models/HighPrecison_4band_1.pth') ##DON'T CHNGE ITERATION-3
    p.add_argument('--pretrained_modelD', '-Da', type=str, default='models/HighPrecison_4band_2.pth') ##DON'T CHNGE ITERATION-4
    p.add_argument('--pretrained_modelE', '-E', type=str, default='models/NewLayer_4band_1.pth') ##DON'T CHNGE ITERATION-5
    p.add_argument('--pretrained_modelF', '-F', type=str, default='models/NewLayer_4band_2.pth') ##DON'T CHNGE ITERATION-6
    p.add_argument('--pretrained_modelG', '-G', type=str, default='models/NewLayer_4band_3.pth') ##DON'T CHNGE ITERATION-7
    p.add_argument('--pretrained_modelH', '-Hm', type=str, default='models/MGM-v5-MIDSIDE-44100-BETA1.pth') ##DON'T CHNGE ITERATION-8
    p.add_argument('--pretrained_modelI', '-Im', type=str, default='models/MGM-v5-MIDSIDE-44100-BETA2.pth') ##DON'T CHNGE ITERATION-9
    p.add_argument('--pretrained_modelJ', '-J', type=str, default='models/MGM-v5-3Band-44100-BETA.pth') ##DON'T CHNGE ITERATION-10
    ##p.add_argument('--pretrained_modelK', '-K', type=str, default='models/MGM-v5-2Band-32000-BETA1.pth') ##DON'T CHNGE ITERATION-11
    ##p.add_argument('--pretrained_modelL', '-L', type=str, default='models/MGM-v5-2Band-32000-BETA2.pth') ##DON'T CHNGE ITERATION-12
    p.add_argument('--pretrained_modelM', '-Mm', type=str, default='models/LOFI_2band-1_33966KB.pth') ##DON'T CHNGE ITERATION-13
    p.add_argument('--pretrained_modelN', '-Nm', type=str, default='models/LOFI_2band-2_33966KB.pth') ##DON'T CHNGE ITERATION-14
    p.add_argument('--deepextraction', '-D', action='store_true')
    p.add_argument('--saveindivsep', '-s', action='store_true')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--nn_architecture', '-n', type=str, default='default') ##DON'T CHNGE ITERATION-1, ITERATION-2, ITERATION-8, ITERATION-9, ITERATION-10, ITERATION-11, ITERATION-12
    p.add_argument('--nn_architectureA', '-aB', type=str, default='123821KB') ##DON'T CHNGE ITERATION-3, ITERATION-4
    p.add_argument('--nn_architectureB', '-bA', type=str, default='129605KB') ##DON'T CHNGE ITERATION-5, ITERATION-6, ITERATION-7
    p.add_argument('--nn_architectureC', '-bC', type=str, default='33966KB') ##DON'T CHNGE ITERATION-13, ITERATION-14
    p.add_argument('--model_params', '-m', type=str, default='modelparams/4band_44100.json') ##DON'T CHNGE ITERATION-1, ITERATION-2, ITERATION-3, ITERATION-4, ITERATION-5, ITERATION-6, ITERATION-7
    p.add_argument('--model_paramsB', '-mB', type=str, default='modelparams/3band_44100_mid.json') ##DON'T CHNGE ITERATION-8, ITERATION-9
    p.add_argument('--model_paramsC', '-mC', type=str, default='modelparams/3band_44100.json') ##DON'T CHNGE ITERATION-10
    p.add_argument('--model_paramsD', '-mD', type=str, default='modelparams/2band_32000.json') ##DON'T CHNGE ITERATION-11, ITERATION-12
    p.add_argument('--model_paramsE', '-mE', type=str, default='modelparams/2band_44100_lofi.json') ##DON'T CHNGE ITERATION-13, ITERATION-14
    p.add_argument('--window_size', '-w', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--high_end_process', '-H', type=str, choices=['none', 'bypass', 'correlation'], default='none')
    p.add_argument('--aggressiveness', '-A', type=float, default=0.05)
    args = p.parse_args()
    
 ####################################################-ITERATION-1-####################################################

    if args.nn_architecture == 'default':
            from lib import nets
    
    dir = 'ensembled/temp'
    for file in os.scandir(dir):
        os.remove(file.path)

    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_params)    
    
    start_time = time.time()

    print('loading MGM-v5-4Band-44100-BETA1...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelA, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelA))[0]
    sf.write(os.path.join('ensembled/temp', '1_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '1_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

 ####################################################-ITERATION-2-####################################################

    if args.nn_architecture == 'default':
            from lib import nets
    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_params)    
    
    start_time = time.time()

    print('loading MGM-v5-4Band-44100-BETA2...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelB, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelB))[0]
    sf.write(os.path.join('ensembled/temp', '2_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '2_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

 ####################################################-ITERATION-3-####################################################

    if args.nn_architectureA == '123821KB':
            from lib import nets_123821KB as nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_params)    
    
    start_time = time.time()

    print('loading HighPrecison_4band_1...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelC, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelC))[0]
    sf.write(os.path.join('ensembled/temp', '3_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '3_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

 ####################################################-ITERATION-4-####################################################

    if args.nn_architectureA == '123821KB':
            from lib import nets_123821KB as nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_params)    
    
    start_time = time.time()

    print('loading HighPrecison_4band_2...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelD, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelD))[0]
    sf.write(os.path.join('ensembled/temp', '4_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '4_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

 ####################################################-ITERATION-5-####################################################

    if args.nn_architectureB == '129605KB':
            from lib import nets_129605KB as nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_params)    
    
    start_time = time.time()

    print('loading NewLayer_4band_1...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelE, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelE))[0]
    sf.write(os.path.join('ensembled/temp', '5_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '5_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

 ####################################################-ITERATION-6-####################################################

    if args.nn_architectureB == '129605KB':
            from lib import nets_129605KB as nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_params)    
    
    start_time = time.time()

    print('loading NewLayer_4band_2...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelF, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelF))[0]
    sf.write(os.path.join('ensembled/temp', '6_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '6_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

 ####################################################-ITERATION-7-####################################################

    if args.nn_architectureB == '129605KB':
            from lib import nets_129605KB as nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_params)    
    
    start_time = time.time()

    print('loading NewLayer_4band_3...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelG, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelG))[0]
    sf.write(os.path.join('ensembled/temp', '7_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '7_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

 ####################################################-ITERATION-8-####################################################

    if args.nn_architecture == 'default':
            from lib import nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_paramsB)    
    
    start_time = time.time()

    print('loading MGM-v5-MIDSIDE-44100-BETA1...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelH, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelH))[0]
    sf.write(os.path.join('ensembled/temp', '8_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '8_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

####################################################-ITERATION-9-####################################################

    if args.nn_architecture == 'default':
            from lib import nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_paramsB)    
    
    start_time = time.time()

    print('loading MGM-v5-MIDSIDE-44100-BETA2...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelI, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelI))[0]
    sf.write(os.path.join('ensembled/temp', '9_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '9_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

####################################################-ITERATION-10-####################################################

    if args.nn_architecture == 'default':
            from lib import nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_paramsC)    
    
    start_time = time.time()

    print('loading MGM-v5-3Band-44100-BETA...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelJ, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelJ))[0]
    sf.write(os.path.join('ensembled/temp', '10_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '10_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

####################################################-ITERATION-11-####################################################

##    if args.nn_architecture == 'default':
##            from lib import nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
##   mp = ModelParameters(args.model_paramsD)    
    
##    start_time = time.time()

##    print('loading MGM-v5-2Band-32000-BETA1...', end=' ')

##    device = torch.device('cpu')
##    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
##    model.load_state_dict(torch.load(args.pretrained_modelK, map_location=device))
##    if torch.cuda.is_available() and args.gpu >= 0:
##        device = torch.device('cuda:{}'.format(args.gpu))
##        model.to(device)

##    print('done')

##    print('loading & stft of wave source...', end=' ')
    
##    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
##    basename = os.path.splitext(os.path.basename(args.input))[0]
##    bands_n = len(mp.param['band'])
             
##    for d in range(bands_n, 0, -1):        
##       bp = mp.param['band'][d]
                 
##        if d == bands_n: # high-end band
##            X_wave[d], _ = librosa.load(
##                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
##                
##            if X_wave[d].ndim == 1:
##                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
##        else: # lower bands
##            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

##        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
##        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
##            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
##            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

##    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
##    del X_wave, X_spec_s
        
##    print('done')
    
##    vr = VocalRemover(model, device, args.window_size)

##    if args.tta:
##        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
##    else:
##        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
##
##    if args.postprocess:
##        print('post processing...', end=' ')
##        pred_inv = np.clip(X_mag - pred, 0, np.inf)
##        pred = spec_utils.mask_silence(pred, pred_inv)
##       print('done')
        
##    if 'is_vocal_model' in mp.param: # swap
##        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
##    else:
##        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
##    print('inverse stft of {}...'.format(stems['inst']), end=' ')
##    y_spec_m = pred * X_phase
##    v_spec_m = X_spec_m - y_spec_m
##   
##    if args.high_end_process == 'bypass':
##        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
##    elif args.high_end_process == 'correlation':       
##        for i in range(input_high_end.shape[2]):            
##            for c in range(2):
##                X_mag_max = np.amax(input_high_end[c, :, i])    
##                b1 = mp.param['pre_filter_start']-input_high_end_h//2
##                b2 = mp.param['pre_filter_start']-1
##                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
##                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
##                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
##           
##        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
##    else:
##        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
##    
##    print('done')
##    model_name = os.path.splitext(os.path.basename(args.pretrained_modelK))[0]
##    sf.write(os.path.join('ensembled/temp', '11_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])
##
##    if True:
##        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
##        #v_spec_m = X_spec_m - y_spec_m
##        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
##        print('done')
##        sf.write(os.path.join('ensembled/temp', '11_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])
##
##    if args.output_image:
##        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
##            image = spec_utils.spectrogram_to_image(y_spec_m)
##            _, bin_image = cv2.imencode('.jpg', image)
##            bin_image.tofile(f)
##        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
##            image = spec_utils.spectrogram_to_image(v_spec_m)
##            _, bin_image = cv2.imencode('.jpg', image)
##            bin_image.tofile(f)
##
##    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

####################################################-ITERATION-12-####################################################

##    if args.nn_architecture == 'default':
##            from lib import nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
##   mp = ModelParameters(args.model_paramsD)    
    
##    start_time = time.time()

##    print('loading MGM-v5-2Band-32000-BETA...', end=' ')

##    device = torch.device('cpu')
##    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
##    model.load_state_dict(torch.load(args.pretrained_modelK, map_location=device))
##    if torch.cuda.is_available() and args.gpu >= 0:
##        device = torch.device('cuda:{}'.format(args.gpu))
##        model.to(device)

##    print('done')

##    print('loading & stft of wave source...', end=' ')
    
##    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
##    basename = os.path.splitext(os.path.basename(args.input))[0]
##    bands_n = len(mp.param['band'])
             
##    for d in range(bands_n, 0, -1):        
##       bp = mp.param['band'][d]
                 
##        if d == bands_n: # high-end band
##            X_wave[d], _ = librosa.load(
##                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
##                
##            if X_wave[d].ndim == 1:
##                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
##        else: # lower bands
##            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

##        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
##        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
##            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
##            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

##    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
##    del X_wave, X_spec_s
        
##    print('done')
    
##    vr = VocalRemover(model, device, args.window_size)

##    if args.tta:
##        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
##    else:
##        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
##
##    if args.postprocess:
##        print('post processing...', end=' ')
##        pred_inv = np.clip(X_mag - pred, 0, np.inf)
##        pred = spec_utils.mask_silence(pred, pred_inv)
##       print('done')
        
##    if 'is_vocal_model' in mp.param: # swap
##        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
##    else:
##        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
##    print('inverse stft of {}...'.format(stems['inst']), end=' ')
##    y_spec_m = pred * X_phase
##    v_spec_m = X_spec_m - y_spec_m
##   
##    if args.high_end_process == 'bypass':
##        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
##    elif args.high_end_process == 'correlation':       
##        for i in range(input_high_end.shape[2]):            
##            for c in range(2):
##                X_mag_max = np.amax(input_high_end[c, :, i])    
##                b1 = mp.param['pre_filter_start']-input_high_end_h//2
##                b2 = mp.param['pre_filter_start']-1
##                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
##                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
##                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
##           
##        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
##    else:
##        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
##    
##    print('done')
##    model_name = os.path.splitext(os.path.basename(args.pretrained_modelK))[0]
##    sf.write(os.path.join('ensembled/temp', '11_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])
##
##    if True:
##        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
##        #v_spec_m = X_spec_m - y_spec_m
##        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
##        print('done')
##        sf.write(os.path.join('ensembled/temp', '11_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])
##
##    if args.output_image:
##        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
##            image = spec_utils.spectrogram_to_image(y_spec_m)
##            _, bin_image = cv2.imencode('.jpg', image)
##            bin_image.tofile(f)
##        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
##            image = spec_utils.spectrogram_to_image(v_spec_m)
##            _, bin_image = cv2.imencode('.jpg', image)
##            bin_image.tofile(f)
##
##    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

####################################################-ITERATION-13-####################################################

    if args.nn_architectureC == '33966KB':
            from lib import nets_33966KB as nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_paramsE)    
    
    start_time = time.time()

    print('loading LOFI_2band-1_33966KB...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelM, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelM))[0]
    sf.write(os.path.join('ensembled/temp', '13_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '13_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

####################################################-ITERATION-14-####################################################

    if args.nn_architectureC == '33966KB':
            from lib import nets_33966KB as nets

    
    #if '' == args.model_params:
    #    mp = ModelParameters(args.pretrained_model)
    #else:
    mp = ModelParameters(args.model_paramsE)    
    
    start_time = time.time()

    print('loading LOFI_2band-2_33966KB...', end=' ')

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
    model.load_state_dict(torch.load(args.pretrained_modelN, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    print('done')

    print('loading & stft of wave source...', end=' ')
    
    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
    basename = os.path.splitext(os.path.basename(args.input))[0]
    bands_n = len(mp.param['band'])
             
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
                
        if d == bands_n: # high-end band
            X_wave[d], _ = librosa.load(
                args.input, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])

        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['reverse'])
        
        if d == bands_n and args.high_end_process in ['bypass', 'correlation']:
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s
        
    print('done')
    
    vr = VocalRemover(model, device, args.window_size)

    if args.tta:
        pred, X_mag, X_phase = vr.inference_tta(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})
    else:
        pred, X_mag, X_phase = vr.inference(X_spec_m, {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']})

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
        print('done')
        
    if 'is_vocal_model' in mp.param: # swap
        stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
    else:
        stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
        
    print('inverse stft of {}...'.format(stems['inst']), end=' ')
    y_spec_m = pred * X_phase
    v_spec_m = X_spec_m - y_spec_m
   
    if args.high_end_process == 'bypass':
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    elif args.high_end_process == 'correlation':       
        for i in range(input_high_end.shape[2]):            
            for c in range(2):
                X_mag_max = np.amax(input_high_end[c, :, i])    
                b1 = mp.param['pre_filter_start']-input_high_end_h//2
                b2 = mp.param['pre_filter_start']-1
                if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                    y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                    input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
            
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
    
    print('done')
    model_name = os.path.splitext(os.path.basename(args.pretrained_modelN))[0]
    sf.write(os.path.join('ensembled/temp', '14_{}_{}.wav'.format(model_name, stems['inst'])), wave, mp.param['sr'])

    if True:
        print('inverse stft of {}...'.format(stems['vocals']), end=' ')
        #v_spec_m = X_spec_m - y_spec_m
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
        print('done')
        sf.write(os.path.join('ensembled/temp', '14_{}_{}.wav'.format(model_name, stems['vocals'])), wave, mp.param['sr'])

    if args.output_image:
        with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(y_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(v_spec_m)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

####################################################ITERATIONS-COMPLETE######################################################

####################################################-ENSEMBLING-BEGIN-#######################################################


    if args.deepextraction:
        print('Ensembling Instrumentals...')
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1_MGM-v5-4Band-44100-BETA1_Instruments.wav ensembled/temp/2_MGM-v5-4Band-44100-BETA2_Instruments.wav -o ensembled/temp/1E2E_ensam1")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/3_HighPrecison_4band_1_Instruments.wav ensembled/temp/4_HighPrecison_4band_2_Instruments.wav -o ensembled/temp/3E4E_ensam1")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/5_NewLayer_4band_1_Instruments.wav ensembled/temp/6_NewLayer_4band_2_Instruments.wav -o ensembled/temp/5E6E_ensam3")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2E_ensam1_v.wav ensembled/temp/3E4E_ensam1_v.wav -o ensembled/temp/1E2E3E4E_ensam4")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2E3E4E_ensam4_v.wav ensembled/temp/5E6E_ensam3_v.wav -o ensembled/temp/A6_ensam5")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/7_NewLayer_4band_3_Instruments.wav ensembled/temp/A6_ensam5_v.wav -o ensembled/temp/1STHALF")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8_MGM-v5-MIDSIDE-44100-BETA1_Instruments.wav ensembled/temp/9_MGM-v5-MIDSIDE-44100-BETA2_Instruments.wav -o ensembled/temp/8E9E_ensam1")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/10_MGM-v5-3Band-44100-BETA_Instruments.wav ensembled/temp/13_LOFI_2band-1_33966KB_Instruments.wav -o ensembled/temp/10E13E_ensam3")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9E_ensam1_v.wav ensembled/temp/10E13E_ensam3_v.wav -o ensembled/temp/8E9E10E13E_ensam4")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9E10E13E_ensam4_v.wav ensembled/temp/14_LOFI_2band-2_33966KB_Instruments.wav -o ensembled/temp/2NDHALF")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1STHALF_v.wav ensembled/temp/2NDHALF_v.wav -o ensembled/temp/Complete")

        print('Ensembling Vocals...')
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1_MGM-v5-4Band-44100-BETA1_Vocals.wav ensembled/temp/2_MGM-v5-4Band-44100-BETA2_Vocals.wav -o ensembled/temp/1E2EV_ensam1")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/3_HighPrecison_4band_1_Vocals.wav ensembled/temp/4_HighPrecison_4band_2_Vocals.wav -o ensembled/temp/3E4EV_ensam1")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/5_NewLayer_4band_1_Vocals.wav ensembled/temp/6_NewLayer_4band_2_Vocals.wav -o ensembled/temp/5E6EV_ensam3")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2EV_ensam1_v.wav ensembled/temp/3E4EV_ensam1_v.wav -o ensembled/temp/1E2E3E4EV_ensam4")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2E3E4EV_ensam4_v.wav ensembled/temp/5E6EV_ensam3_v.wav -o ensembled/temp/A6V_ensam5")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/7_NewLayer_4band_3_Vocals.wav ensembled/temp/A6V_ensam5_v.wav -o ensembled/temp/1STHALFV")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8_MGM-v5-MIDSIDE-44100-BETA1_Vocals.wav ensembled/temp/9_MGM-v5-MIDSIDE-44100-BETA2_Vocals.wav -o ensembled/temp/8E9EV_ensam1")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/10_MGM-v5-3Band-44100-BETA_Vocals.wav ensembled/temp/13_LOFI_2band-1_33966KB_Vocals.wav -o ensembled/temp/10E13EV_ensam3")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9EV_ensam1_v.wav ensembled/temp/10E13EV_ensam3_v.wav -o ensembled/temp/8E9E10E13EV_ensam4")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9E10E13EV_ensam4_v.wav ensembled/temp/14_LOFI_2band-2_33966KB_Vocals.wav -o ensembled/temp/2NDHALFV")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1STHALFV_v.wav ensembled/temp/2NDHALFV_v.wav -o ensembled/temp/CompleteV")

        print('Performing Deep Extraction...')
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/Complete_v.wav ensembled/temp/CompleteV_v.wav -o ensembled/temp/difftemp")
        os.system("python lib/spec_utils.py -a invertB -m modelparams/1band_sr44100_hl512.json ensembled/temp/Complete_v.wav ensembled/temp/difftemp_v.wav -o ensembled/temp/difftempC")
        os.rename('ensembled/temp/difftempC_v.wav', 'ensembled/{}_ALLMODELS_Ensembled_DeepExtraction_Instrumental.wav'.format(basename))
        os.rename('ensembled/temp/Complete_v.wav', 'ensembled/{}_ALLMODELS_Ensembled_Instrumental.wav'.format(basename))
        os.rename('ensembled/temp/CompleteV_v.wav', 'ensembled/{}_ALLMODELS_Ensembled_Vocals.wav'.format(basename))
        print('Deep Extraction Complete!')
    else:
        print('Ensembling Instrumentals...')
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1_MGM-v5-4Band-44100-BETA1_Instruments.wav ensembled/temp/2_MGM-v5-4Band-44100-BETA2_Instruments.wav -o ensembled/temp/1E2E_ensam1")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/3_HighPrecison_4band_1_Instruments.wav ensembled/temp/4_HighPrecison_4band_2_Instruments.wav -o ensembled/temp/3E4E_ensam1")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/5_NewLayer_4band_1_Instruments.wav ensembled/temp/6_NewLayer_4band_2_Instruments.wav -o ensembled/temp/5E6E_ensam3")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2E_ensam1_v.wav ensembled/temp/3E4E_ensam1_v.wav -o ensembled/temp/1E2E3E4E_ensam4")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2E3E4E_ensam4_v.wav ensembled/temp/5E6E_ensam3_v.wav -o ensembled/temp/A6_ensam5")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/7_NewLayer_4band_3_Instruments.wav ensembled/temp/A6_ensam5_v.wav -o ensembled/temp/1STHALF")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8_MGM-v5-MIDSIDE-44100-BETA1_Instruments.wav ensembled/temp/9_MGM-v5-MIDSIDE-44100-BETA2_Instruments.wav -o ensembled/temp/8E9E_ensam1")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/10_MGM-v5-3Band-44100-BETA_Instruments.wav ensembled/temp/13_LOFI_2band-1_33966KB_Instruments.wav -o ensembled/temp/10E13E_ensam3")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9E_ensam1_v.wav ensembled/temp/10E13E_ensam3_v.wav -o ensembled/temp/8E9E10E13E_ensam4")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9E10E13E_ensam4_v.wav ensembled/temp/14_LOFI_2band-2_33966KB_Instruments.wav -o ensembled/temp/2NDHALF")
        os.system("python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1STHALF_v.wav ensembled/temp/2NDHALF_v.wav -o ensembled/temp/Complete")
        os.rename('ensembled/temp/Complete_v.wav', 'ensembled/{}_ALLMODELS_Ensembled_Instrumental.wav'.format(basename))

        print('Ensembling Vocals...')
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1_MGM-v5-4Band-44100-BETA1_Vocals.wav ensembled/temp/2_MGM-v5-4Band-44100-BETA2_Vocals.wav -o ensembled/temp/1E2EV_ensam1")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/3_HighPrecison_4band_1_Vocals.wav ensembled/temp/4_HighPrecison_4band_2_Vocals.wav -o ensembled/temp/3E4EV_ensam1")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/5_NewLayer_4band_1_Vocals.wav ensembled/temp/6_NewLayer_4band_2_Vocals.wav -o ensembled/temp/5E6EV_ensam3")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2EV_ensam1_v.wav ensembled/temp/3E4EV_ensam1_v.wav -o ensembled/temp/1E2E3E4EV_ensam4")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1E2E3E4EV_ensam4_v.wav ensembled/temp/5E6EV_ensam3_v.wav -o ensembled/temp/A6V_ensam5")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/7_NewLayer_4band_3_Vocals.wav ensembled/temp/A6V_ensam5_v.wav -o ensembled/temp/1STHALFV")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8_MGM-v5-MIDSIDE-44100-BETA1_Vocals.wav ensembled/temp/9_MGM-v5-MIDSIDE-44100-BETA2_Vocals.wav -o ensembled/temp/8E9EV_ensam1")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/10_MGM-v5-3Band-44100-BETA_Vocals.wav ensembled/temp/13_LOFI_2band-1_33966KB_Vocals.wav -o ensembled/temp/10E13EV_ensam3")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9EV_ensam1_v.wav ensembled/temp/10E13EV_ensam3_v.wav -o ensembled/temp/8E9E10E13EV_ensam4")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/8E9E10E13EV_ensam4_v.wav ensembled/temp/14_LOFI_2band-2_33966KB_Vocals.wav -o ensembled/temp/2NDHALFV")
        os.system("python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json ensembled/temp/1STHALFV_v.wav ensembled/temp/2NDHALFV_v.wav -o ensembled/temp/CompleteV")
        os.rename('ensembled/temp/CompleteV_v.wav', 'ensembled/{}_ALLMODELS_Ensembled_Vocals.wav'.format(basename))

    if args.saveindivsep:
        print('Saving Individual Separations...')
        os.rename('ensembled/temp/1_MGM-v5-4Band-44100-BETA1_Instruments.wav', 'separated/{}_MGM-v5-4Band-44100-BETA1_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/2_MGM-v5-4Band-44100-BETA2_Instruments.wav', 'separated/{}_MGM-v5-4Band-44100-BETA2_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/3_HighPrecison_4band_1_Instruments.wav', 'separated/{}_HighPrecison_4band_1_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/4_HighPrecison_4band_2_Instruments.wav', 'separated/{}_HighPrecison_4band_2_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/5_NewLayer_4band_1_Instruments.wav', 'separated/{}_NewLayer_4band_1_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/6_NewLayer_4band_2_Instruments.wav', 'separated/{}_NewLayer_4band_2_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/7_NewLayer_4band_3_Instruments.wav', 'separated/{}_NewLayer_4band_3_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/1_MGM-v5-4Band-44100-BETA1_Vocals.wav', 'separated/{}_MGM-v5-4Band-44100-BETA1_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/2_MGM-v5-4Band-44100-BETA2_Vocals.wav', 'separated/{}_MGM-v5-4Band-44100-BETA2_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/3_HighPrecison_4band_1_Vocals.wav', 'separated/{}_HighPrecison_4band_1_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/4_HighPrecison_4band_2_Vocals.wav', 'separated/{}_HighPrecison_4band_2_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/5_NewLayer_4band_1_Vocals.wav', 'separated/{}_NewLayer_4band_1_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/6_NewLayer_4band_2_Vocals.wav', 'separated/{}_NewLayer_4band_2_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/7_NewLayer_4band_3_Vocals.wav', 'separated/{}_NewLayer_4band_3_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/8_MGM-v5-MIDSIDE-44100-BETA1_Instruments.wav', 'separated/{}_MGM-v5-MIDSIDE-44100-BETA1_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/9_MGM-v5-MIDSIDE-44100-BETA2_Instruments.wav', 'separated/{}_MGM-v5-MIDSIDE-44100-BETA2_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/10_MGM-v5-3Band-44100-BETA_Instruments.wav', 'separated/{}_MGM-v5-3Band-44100-BETA_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/13_LOFI_2band-1_33966KB_Instruments.wav', 'separated/{}_LOFI_2band-_33966KB_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/14_LOFI_2band-2_33966KB_Instruments.wav', 'separated/{}_LOFI_2band-2_33966KB_Instruments.wav'.format(basename))
        os.rename('ensembled/temp/8_MGM-v5-MIDSIDE-44100-BETA1_Vocals.wav', 'separated/{}_MGM-v5-MIDSIDE-44100-BETA1_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/9_MGM-v5-MIDSIDE-44100-BETA2_Vocals.wav', 'separated/{}_MGM-v5-MIDSIDE-44100-BETA2_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/10_MGM-v5-3Band-44100-BETA_Vocals.wav', 'separated/{}_MGM-v5-3Band-44100-BETA_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/13_LOFI_2band-1_33966KB_Vocals.wav', 'separated/{}_LOFI_2band-1_33966KB_Vocals.wav'.format(basename))
        os.rename('ensembled/temp/14_LOFI_2band-2_33966KB_Vocals.wav', 'separated/{}_LOFI_2band-2_33966KB_Vocals.wav'.format(basename))
        os.remove("ensembled/temp/A6V_ensam5_v.wav")
        os.remove("ensembled/temp/1E2E3E4EV_ensam4_v.wav")
        os.remove("ensembled/temp/5E6EV_ensam3_v.wav")
        os.remove("ensembled/temp/3E4EV_ensam1_v.wav")
        os.remove("ensembled/temp/1E2EV_ensam1_v.wav")
        os.remove("ensembled/temp/A6_ensam5_v.wav")
        os.remove("ensembled/temp/1E2E3E4E_ensam4_v.wav")
        os.remove("ensembled/temp/5E6E_ensam3_v.wav")
        os.remove("ensembled/temp/3E4E_ensam1_v.wav")
        os.remove("ensembled/temp/1E2E_ensam1_v.wav")
        os.remove("ensembled/temp/1STHALF_v.wav")
        os.remove("ensembled/temp/1STHALFV_v.wav")
        os.remove("ensembled/temp/2NDHALF_v.wav")
        os.remove("ensembled/temp/2NDHALFV_v.wav")
        os.remove("ensembled/temp/8E9E_ensam1_v.wav")
        os.remove("ensembled/temp/8E9E10E13E_ensam4_v.wav")
        os.remove("ensembled/temp/8E9E10E13EV_ensam4_v.wav")
        os.remove("ensembled/temp/8E9EV_ensam1_v.wav")
        os.remove("ensembled/temp/10E13E_ensam3_v.wav")
        os.remove("ensembled/temp/10E13EV_ensam3_v.wav")
        print('Complete!')
    else:
        print('Cleaning Up...')
        dir = 'ensembled/temp'
        for file in os.scandir(dir):
            os.remove(file.path)
        print('Complete!')

if __name__ == '__main__':
    main()
