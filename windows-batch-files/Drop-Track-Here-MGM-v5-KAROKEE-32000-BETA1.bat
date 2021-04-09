@echo off
 
set model=MGM-v5-KAROKEE-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
 
pause