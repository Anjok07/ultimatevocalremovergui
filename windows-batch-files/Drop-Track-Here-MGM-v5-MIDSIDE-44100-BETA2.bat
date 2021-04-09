@echo off
 
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m 3band_44100_mid.json -w 352 -P models\%model%.pth -t -i %1
 
pause