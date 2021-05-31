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
    p.add_argument('--is_vocal_model', '-vm', action='store_true')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--nn_architecture', '-n', type=str, default='default')
    p.add_argument('--window_size', '-w', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--high_end_process', '-H', type=str, choices=['none', 'bypass', 'correlation', 'mirroring', 'mirroring2'], default='none')
    p.add_argument('--aggressiveness', '-A', type=float, default=0.05)
    p.add_argument('--savein', '-s', action='store_true')
    p.add_argument('--model_params', '-m', type=str, default='modelparams/4band_44100.json')
    p.add_argument('--model_paramsB', type=str, default='modelparams/3band_44100_mid.json') 
    p.add_argument('--model_paramsC', type=str, default='modelparams/3band_44100.json')
    p.add_argument('--model_paramsD', type=str, default='modelparams/2band_32000.json') 
    p.add_argument('--model_paramsE', type=str, default='modelparams/2band_44100_lofi.json') 
    p.add_argument('--pretrained_model', '-P', type=str, default='models/MGM-v5-4Band-44100-BETA1.pth') 
    p.add_argument('--pretrained_modelB', type=str, default='models/MGM-v5-4Band-44100-BETA2.pth') 
    p.add_argument('--pretrained_modelC', type=str, default='models/HighPrecison_4band_1.pth')
    p.add_argument('--pretrained_modelD', type=str, default='models/HighPrecison_4band_2.pth') 
    p.add_argument('--pretrained_modelE', type=str, default='models/NewLayer_4band_1.pth')
    p.add_argument('--pretrained_modelF', type=str, default='models/NewLayer_4band_2.pth')
    p.add_argument('--pretrained_modelG', type=str, default='models/NewLayer_4band_3.pth') 
    p.add_argument('--pretrained_modelH', type=str, default='models/MGM-v5-MIDSIDE-44100-BETA1.pth')
    p.add_argument('--pretrained_modelI', type=str, default='models/MGM-v5-MIDSIDE-44100-BETA2.pth') 
    p.add_argument('--pretrained_modelJ', type=str, default='models/MGM-v5-3Band-44100-BETA.pth')
    p.add_argument('--pretrained_modelK', type=str, default='models/MGM-v5-2Band-32000-BETA1.pth')
    p.add_argument('--pretrained_modelL', type=str, default='models/MGM-v5-2Band-32000-BETA2.pth') 
    p.add_argument('--pretrained_modelM', type=str, default='models/LOFI_2band-1_33966KB.pth') 
    p.add_argument('--pretrained_modelN', type=str, default='models/LOFI_2band-2_33966KB.pth')

    args = p.parse_args()
    
####################################################-CLEAR-TEMP-FOLDER-########################################
    dir = 'ensembled/temp'
    for file in os.scandir(dir):
        os.remove(file.path)
####################################################-LOOPS-####################################################

    loops = [
        {
            'model_name':'MGM-v5-4Band-44100-BETA1',
            'model_location':args.pretrained_model,
            'model_params':args.model_params,
            'using_archtecture': 'default',
        },
        {
            'model_name':'MGM-v5-4Band-44100-BETA2',
            'model_location':args.pretrained_modelB,
            'model_params':args.model_params,
            'using_archtecture': 'default'
        },
        {
            'model_name':'HighPrecison_4band_1',
            'model_location':args.pretrained_modelC,
            'model_params':args.model_params,
            'using_archtecture': '123821KB'
        },
        {
            'model_name':'HighPrecison_4band_2',
            'model_location':args.pretrained_modelD,
            'model_params':args.model_params,
            'using_archtecture': '123821KB'
        },
        {
            'model_name':'NewLayer_4band_1',
            'model_location':args.pretrained_modelE,
            'model_params':args.model_params,
            'using_archtecture': '129605KB'
        },
        {
            'model_name':'NewLayer_4band_2',
            'model_location':args.pretrained_modelF,
            'model_params':args.model_params,
            'using_archtecture': '129605KB'
        },
        {
            'model_name':'NewLayer_4band_3',
            'model_location':args.pretrained_modelG,
            'model_params':args.model_params,
            'using_archtecture': '129605KB'
        },
        {
            'model_name':'MGM-v5-MIDSIDE-44100-BETA1',
            'model_location':args.pretrained_modelH,
            'model_params':args.model_params,
            'using_archtecture': 'default'
        },
        {
            'model_name':'MGM-v5-MIDSIDE-44100-BETA2',
            'model_location':args.pretrained_modelI,
            'model_params':args.model_params,
            'using_archtecture': 'default'
        },
        {
            'model_name':'MGM-v5-4Band-44100-BETA2',
            'model_location':args.pretrained_modelJ,
            'model_params':args.model_params,
            'using_archtecture': 'default'
        },
        {
            'model_name':'MGM-v5-3Band-44100-BETA',
            'model_location':args.pretrained_modelK,
            'model_params':args.model_params,
            'using_archtecture': 'default'
        },
        {
            'model_name':'MGM-v5-2Band-32000-BETA1',
            'model_location':args.pretrained_modelL,
            'model_params':args.model_params,
            'using_archtecture': 'default'
        },
        {
            'model_name':'LOFI_2band-1_33966KB',
            'model_location':args.pretrained_modelM,
            'model_params':args.model_params,
            'using_archtecture': '33966KB'
        },
        {
            'model_name':'LOFI_2band-2_33966KB',
            'model_location':args.pretrained_modelN,
            'model_params':args.model_params,
            'using_archtecture': '33966KB'
        }
    ]

    from tqdm.auto import tqdm

    for i, c in tqdm(enumerate(loops), disable=True, desc='Iterations..'):
        arch_now = c['using_archtecture']

        if arch_now == 'default':
            from lib import nets
        elif arch_now == '33966KB':
            from lib import nets_33966KB as nets
        elif arch_now == '123821KB':
            from lib import nets_123821KB as nets
        elif arch_now == '129605KB':
            from lib import nets_129605KB as nets
        elif arch_now == '537238KB':
            from lib import nets_537238KB as nets
        elif arch_now == '2064829KB':
            from lib import nets_2064829KB as nets

        
        #if '' == args.model_params:
        #    mp = ModelParameters(args.pretrained_model)
        #else:
        mp = ModelParameters(c['model_params'])    
        
        start_time = time.time() 

        print('loading {}...'.format(c['model_location']), end=' ') 

        device = torch.device('cpu')
        model = nets.CascadedASPPNet(mp.param['bins'] * 2)
        model.load_state_dict(torch.load(c['model_location'], map_location=device))
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
            
            if d == bands_n and args.high_end_process != 'none':
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

        if args.postprocess: # if c['do_postprocess']:
            print('post processing...', end=' ')
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
            print('done')
            
        if 'is_vocal_model' in mp.param or args.is_vocal_model: 
            stems = {'inst': 'Vocals', 'vocals': 'Instruments'}
        else:
            stems = {'inst': 'Instruments', 'vocals': 'Vocals'}
            
        print('inverse stft of {}...'.format(stems['inst']), end=' ')
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m
    
        if args.high_end_process == 'bypass':
            wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
        elif args.high_end_process == 'correlation': 
            print('Deprecated: correlation will be removed in the final release. Please use the mirroring instead.')
            
            for i in range(input_high_end.shape[2]):            
                for c in range(2):
                    X_mag_max = np.amax(input_high_end[c, :, i])    
                    b1 = mp.param['pre_filter_start']-input_high_end_h//2
                    b2 = mp.param['pre_filter_start']-1
                    if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                        y_mag = np.median(y_spec_m[c, b1:b2, i])                       
                        input_high_end[c, :, i] = np.true_divide(input_high_end[c, :, i], abs(X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))
                
            wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end)        
        elif args.high_end_process.startswith('mirroring'):        
            input_high_end_ = spec_utils.mirroring(args.high_end_process, y_spec_m, input_high_end, mp)
            
            wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)       
        else:
            wave = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
        
        print('done')
        model_name = os.path.splitext(os.path.basename(c['model_location']))[0]
        sf.write(os.path.join('ensembled/temp', f"{i+1}_{model_name}_{stems['inst']}.wav"), wave, mp.param['sr'])
        if args.savein:
            sf.write(os.path.join('separated', f"{basename}_{model_name}_{stems['inst']}.wav"), wave, mp.param['sr'])

        if True:
            print('inverse stft of {}...'.format(stems['vocals']), end=' ')

            if args.high_end_process.startswith('mirroring'):        
                input_high_end_ = spec_utils.mirroring(args.high_end_process, v_spec_m, input_high_end, mp)

                wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp, input_high_end_h, input_high_end_)       
            else:        
                wave = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
            print('done')
            sf.write(os.path.join('ensembled/temp', f"{i+1}_{model_name}_{stems['vocals']}.wav"), wave, mp.param['sr'])
            if args.savein:
                sf.write(os.path.join('separated', f"{basename}_{model_name}_{stems['vocals']}.wav"), wave, mp.param['sr'])


        # if args.output_image:
        #     with open('{}_{}.jpg'.format(basename, stems['inst']), mode='wb') as f:
        #         image = spec_utils.spectrogram_to_image(y_spec_m)
        #         _, bin_image = cv2.imencode('.jpg', image)
        #         bin_image.tofile(f)
        #     with open('{}_{}.jpg'.format(basename, stems['vocals']), mode='wb') as f:
        #         image = spec_utils.spectrogram_to_image(v_spec_m)
        #         _, bin_image = cv2.imencode('.jpg', image)
        #         bin_image.tofile(f)

        # print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

#############################################-ENSEMBLING-BEGIN-################################################
    
    def get_files(folder="", suffix=""):
        return [f"{folder}{i}" for i in os.listdir(folder) if i.endswith(suffix)]
    
    ensembles = [
        {
            'algorithm':'min_mag',
            'model_params':'modelparams/1band_sr44100_hl512.json',
            'files':get_files(folder="ensembled/temp/", suffix="_Instruments.wav"),
            'output':'{}_Ensembled_Instruments'.format(basename)
        },
        {
            'algorithm':'max_mag',
            'model_params':'modelparams/1band_sr44100_hl512.json',
            'files':get_files(folder="ensembled/temp/", suffix="_Vocals.wav"),
            'output': '{}_Ensembled_Vocals'.format(basename)
        }
    ]

    for i,e in tqdm(enumerate(ensembles), desc="Ensembling..."):
        os.system(f"python lib/spec_utils.py -a {e['algorithm']} -m {e['model_params']} {' '.join(e['files'])} -o {e['output']}")
     
     
    dir = 'ensembled/temp'
    for file in os.scandir(dir):
        os.remove(file.path)
    print('Complete!')
    
if __name__ == '__main__':
    main()
