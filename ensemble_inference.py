import argparse
import os, glob

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch

import time, re

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
    #p.add_argument('--is_vocal_model', '-vm', action='store_true')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--window_size', '-w', type=int, default=512)
    #p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--deepextraction', '-D', action='store_true')
    p.add_argument('--aggressiveness', '-A', type=float, default=0.05)
    p.add_argument('--savein', '-s', action='store_true')
    #p.add_argument('--model_params', '-m', type=str, default='modelparams/4band_44100.json')
    dm = [
        'HP_4BAND_3090', 'HP2-4BAND-3090_4band_1', 'HP2-4BAND-3090_4band_2'
    ]
    p.add_argument('-P','--pretrained_models', nargs='+', type=str, default=dm)

    args = p.parse_args()
    
    ensembled_dir = os.path.join("ensembled", "temp")
    basename = os.path.splitext(os.path.basename(args.input))[0]
    basenameb = '{}'.format(os.path.splitext(os.path.basename(args.input))[0])
    
#CLEAR-TEMP-FOLDER
    for file in os.scandir(ensembled_dir):
        os.remove(file.path)

#LOOPS
    models = {
        '^(HP_|HP2-)':
            {
                'model_params': '4band_44100',
            },   
        '^HighPrecison_4band_[12]':
            {
                'model_params': '4band_44100',
            }, 
        'NewLayer_4band_[123]':
            {
                'model_params': '4band_44100',
            },
        'MGM-v5-MIDSIDE-44100-BETA[12]':
            {
                'model_params': '3band_44100_mid',
            },
        'MGM-v5-3Band-44100-BETA':
            {
                'model_params': '3band_44100',
            },
        'MGM-v5-2Band-32000-BETA[12]':
            {
                'model_params': '2band_48000',
            },
        'LOFI_2band-[12]_33966KB':
            {
                'model_params': '2band_44100_lofi',
            },
        'MGM-v5-KAROKEE-32000-BETA1':
            {
                'model_params': '2band_48000',
            },
        'MGM-v5-KAROKEE-32000-BETA2-AGR':
            {
                'model_params': '2band_48000',
            },
        'MGM-v5-Vocal_2Band-32000-BETA[12]':
            {
                'model_params': '2band_48000',
                'is_vocal_model': 'true'
            }
    }

    from tqdm.auto import tqdm

    for ii, model_name in tqdm(enumerate(args.pretrained_models), disable=True, desc='Iterations..'):
        c = {}
               
        for p in models:
            if re.match(p, model_name):
                c = models[p]
                break

        os.system('python inference.py -mt -g {} -m {} -P {} -A {} -w {} {} {} -o {} -i "{}"'.format(
                args.gpu,
                os.path.join('modelparams', c['model_params']) + '.json',
                os.path.join('models', model_name + '.pth'),
                args.aggressiveness,
                args.window_size,
                ('', '-p')[args.postprocess],
                ('', '-t')[args.tta],
                ensembled_dir,
                args.input
            )
        )
        

        # print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))

#ENSEMBLING-BEGIN
    
    def get_files(folder="", suffix=""):
        return [os.path.join(folder, i) for i in os.listdir(folder) if i.endswith(suffix)]

    ensambles = [
        {
            'algorithm':'min_mag',
            'model_params':'modelparams/1band_sr44100_hl512.json',
            'files':get_files(folder=ensembled_dir, suffix="_Instruments.wav"),
            'output':'{}_Ensembled_Instruments'.format(basename)
        },
        {
            'algorithm':'max_mag',
            'model_params':'modelparams/1band_sr44100_hl512.json',
            'files':get_files(folder=ensembled_dir, suffix="_Vocals.wav"),
            'output': '{}_Ensembled_Vocals'.format(basename)
        }
    ]

    for i,e in tqdm(enumerate(ensambles), desc="Ensembling..."):
        os.system("python " + os.path.join("lib", "spec_utils.py") + f" -a {e['algorithm']} -m {e['model_params']} {' '.join(e['files'])} -o {e['output']}")
        
    if args.savein:
        for pm in args.pretrained_models:
            os.rename(
                os.path.join(ensembled_dir, f"{basename}_{pm}_Instruments.wav"),
                os.path.join('separated', f"{basename}_{pm}_Instruments.wav")
            )
            
            os.rename(
                os.path.join(ensembled_dir, f"{basename}_{pm}_Vocals.wav"),
                os.path.join('separated', f"{basename}_{pm}_Vocals.wav")
            )

    if args.deepextraction:
        #def get_files(folder="", files=""):
        #    return [os.path.join(folder, i) for i in os.listdir(folder) if i.endswith(suffix)]
    
        deepext = [
            {
                'algorithm':'deep',
                'model_params':'modelparams/1band_sr44100_hl512.json',
                'file1':"ensembled/{}_Ensembled_Vocals.wav".format(basename),
                'file2':"ensembled/{}_Ensembled_Instruments.wav".format(basename),
                'output':'ensembled/{}_Ensembled_Deep_Extraction'.format(basename)
            }
        ]

        for i,e in tqdm(enumerate(deepext), desc="Performing Deep Extraction..."):
            os.system(f"python lib/spec_utils.py -a {e['algorithm']} -m {e['model_params']} {e['file1']} {e['file2']} -o {e['output']}")
    
    for file in os.scandir(ensembled_dir):
        os.remove(file.path)
    print('Complete!')
    
if __name__ == '__main__':
    main()
