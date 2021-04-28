@echo off

cls
:start
ECHO.
ECHO 1. MGM-v5-2Band-32000-BETA1
ECHO 2. MGM-v5-2Band-32000-BETA2
ECHO 3. MGM-v5-3Band-44100-BETA
ECHO 4. MGM-v5-4Band-44100-BETA1
ECHO 5. MGM-v5-4Band-44100-BETA2
ECHO 6. MGM-v5-KAROKEE-32000-BETA1
ECHO 7. MGM-v5-KAROKEE-32000-BETA2-AGR
ECHO 8. MGM-v5-MIDSIDE-44100-BETA1
ECHO 9. MGM-v5-MIDSIDE-44100-BETA2
ECHO 10. MGM-v5-Vocal_2Band-32000-BETA1
ECHO 11. MGM-v5-Vocal_2Band-32000-BETA2
set choice=
set /p choice=Type the number associated with the model you would like to run and hit 'Enter': 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='1' goto model1
if '%choice%'=='2' goto model2
if '%choice%'=='3' goto model3
if '%choice%'=='4' goto model4
if '%choice%'=='5' goto model5
if '%choice%'=='6' goto model6
if '%choice%'=='7' goto model7
if '%choice%'=='8' goto model8
if '%choice%'=='9' goto model9
if '%choice%'=='10' goto model10
if '%choice%'=='11' goto model11
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:model1
ECHO
set model=MGM-v5-2Band-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model2
ECHO
set model=MGM-v5-2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model3
ECHO
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m 3band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model4
ECHO
set model=MGM-v5-4Band-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m 4band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model5
ECHO
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m 4band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model6
ECHO
set model=MGM-v5-KAROKEE-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model7
ECHO
set model=MGM-v5-KAROKEE-32000-BETA2-AGR
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model8
ECHO
set model=MGM-v5-MIDSIDE-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m 3band_44100_mid.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model9
ECHO
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m 3band_44100_mid.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model10
ECHO
set model=MGM-v5-Vocal_2Band-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model11
ECHO
set model=MGM-v5-Vocal_2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m 2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:end
pause