@echo off
 
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m 4band_44100.json -w 352 -P models\%model%.pth -t -i %1
 
pause