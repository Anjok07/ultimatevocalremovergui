import os
import sys
import random
import glob
import soundfile as sf
import librosa
import numpy as np

def align(file1, file2, file2_aligned, file_subtracted):
  def get_diff(a, b):
    corr = np.correlate(a, b, "full")
    diff = corr.argmax() - (b.shape[0] - 1)
    return diff
  
  def get_diff_val(a, b):
    d = np.abs(a - b)
    return np.mean(d) / max(min(np.max(d), 0.1), 1e-6)
  
  print("Processing: {} / {}".format(file1, file2))
  
  # read tracks
  if os.path.splitext(file1)[1] == '.mp3':        # soundfile doesn't support (?) mp3
    wav1, sr1 = librosa.load(file1, sr=None, mono=False)
    wav2, sr2 = librosa.load(file2, sr=None, mono=False)
    wav1 = wav1.transpose()
    wav2 = wav2.transpose()
  else:
    wav1, sr1 = sf.read(file1, dtype='float32')
    wav2, sr2 = sf.read(file2, dtype='float32')
  assert(sr1 == sr2)
  print("shape {} / {}".format(wav1.shape, wav2.shape))
  wav2_org = wav2.copy()
  
  # pick random position and get diff
  counts = {}       # counting up for each diff value
  for i in range(64):
    index = int(random.uniform(44100 * 2, min(wav1.shape[0], wav2.shape[0]) - 44100 * 2))
    shift = int(random.uniform(-22050,+22050))
    samp1 = wav1[index      :index      +44100, 0]          # currently use left channel
    samp2 = wav2[index+shift:index+shift+44100, 0]
    diff = get_diff(samp1, samp2)
    diff -= shift
    if abs(diff) < 22050:
      if not diff in counts:
        counts[diff] = 0
      counts[diff] += 1
  
  # use max counted diff value
  max_count = 0
  est_diff  = 0
  for diff in counts.keys():
    if counts[diff] > max_count:
      max_count = counts[diff]
      est_diff = diff
  print("estimated diff is {} (count: {})".format(est_diff, max_count))
  
  # make aligned track 2
  if est_diff > 0:
    wav2_aligned = np.append(np.zeros((est_diff, 2)), wav2_org, axis=0)
    sf.write(file2_aligned, wav2_aligned, sr2, subtype='PCM_24')
    min_len = min(wav1.shape[0], wav2_aligned.shape[0])
    wav_sub = wav1[:min_len] - wav2_aligned[:min_len]
    wav_sub = np.clip(wav_sub, -1, +1)
    sf.write(file_subtracted, wav_sub, sr2, subtype='PCM_24')
    print("Audio files have been Aligned! Saving aligned track and difference as {} and {}".format(file2_aligned, file_subtracted))
  elif est_diff < 0:
    wav2_aligned = wav2_org[-est_diff:]
    sf.write(file2_aligned, wav2_aligned, sr2, subtype='PCM_24')
    min_len = min(wav1.shape[0], wav2_aligned.shape[0])
    wav_sub = wav1[:min_len] - wav2_aligned[:min_len]
    wav_sub = np.clip(wav_sub, -1, +1)
    sf.write(file_subtracted, wav_sub, sr2, subtype='PCM_24')  
    print("Audio files have been Aligned! Saving aligned track and difference as {} and {}".format(file2_aligned, file_subtracted))
  elif est_diff == 0:
    min_len = min(wav1.shape[0], wav2.shape[0])
    wav_sub = wav1[:min_len] - wav2[:min_len]
    wav_sub = np.clip(wav_sub, -1, +1)
    sf.write(file_subtracted, wav_sub, sr2, subtype='PCM_24')
    print("Already aligned! - Saving difference as {}".format(file_subtracted))
  
def align_files(pat1, pat2):
  files1 = glob.glob(pat1)
  files2 = glob.glob(pat2)
  files1.sort()
  files2.sort()
  for file1, file2 in zip(files1, files2):
    base_name = os.path.basename(os.path.splitext(file2)[0])
    aligned_file    = base_name + '_aligned.wav'
    subtracted_file = base_name + '_inverted.wav'
    align(file1, file2, aligned_file, subtracted_file)
    
if __name__ == '__main__':
  args = sys.argv
  
  if len(args) == 3:
    align_files(args[1], args[2])
  elif len(args) == 5:
    align(args[1], args[2], args[3], args[4])
  else:
    print("align two tracks\n:" +
          "    python align_tracks.py file-1 file-2 aligned-file-2-to-save subtracted-file-to-save" +
          "align all track\n:" +
          "    python align_tracks.py pattern-1 pattern-2\n" +
          "        saved to '*_aligned.wav' and '*_sub.wav' in the current dir.")
