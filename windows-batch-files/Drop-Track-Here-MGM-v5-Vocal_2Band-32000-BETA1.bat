@echo off
 
set model=MGM-v5-Vocal_2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
 
pause