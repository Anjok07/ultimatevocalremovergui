@echo off
 
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m 3band_44100.json -w 352 -P models\%model%.pth -t -i %1
 
pause