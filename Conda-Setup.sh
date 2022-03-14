cat >environment.yml << END
name: uvr5b
dependencies:
  - python
  - pip
  - Pillow
  - tqdm=4.45.0
  - librosa=0.8.0
  - numba=0.52
  - numpy=1.19.5
  - pysoundfile
  - resampy=0.2.2
  - pip:
      - opencv-python
      - soundstretch
      - samplerate
END
conda env create -n "$1" -f environment.yml
conda activate "$1"
pip install torch==1.8.1 torchvision==0.10.0 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda deactivate
echo "Done."
