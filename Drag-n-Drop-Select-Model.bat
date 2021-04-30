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
ECHO a. MGM-v5-Vocal_2Band-32000-BETA1
ECHO b. MGM-v5-Vocal_2Band-32000-BETA2
ECHO c. LOFI_2band_iter5_2
ECHO d. HighPrecison_4band_1
ECHO e. HighPrecison_4band_2
ECHO f. BigLayer_4band_1.pth
ECHO g. BigLayer_4band_2.pth
ECHO h. BigLayer_4band_3.pth
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
if '%choice%'=='a' goto model10
if '%choice%'=='b' goto model11
if '%choice%'=='c' goto model12
if '%choice%'=='d' goto model13
if '%choice%'=='e' goto model14
if '%choice%'=='f' goto model15
if '%choice%'=='g' goto model16
if '%choice%'=='h' goto model17
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:model1
ECHO
set model=MGM-v5-2Band-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model2
ECHO
set model=MGM-v5-2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model3
ECHO
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model4
ECHO
set model=MGM-v5-4Band-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model5
ECHO
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model6
ECHO
set model=MGM-v5-KAROKEE-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model7
ECHO
set model=MGM-v5-KAROKEE-32000-BETA2-AGR
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model8
ECHO
set model=MGM-v5-MIDSIDE-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model9
ECHO
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model10
ECHO
set model=MGM-v5-Vocal_2Band-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model11
ECHO
set model=MGM-v5-Vocal_2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model12
ECHO
set model=LOFI_2band_iter5_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -w 352 -n 33966KB -P models\%model%.pth -t -i %1
goto end
:model13
ECHO
set model=HighPrecison_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -t -i %1
goto end
:model14
ECHO
set model=HighPrecison_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -t -i %1
goto end
:model15
ECHO
set model=BigLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1
goto end
:model16
ECHO
set model=BigLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1
goto end
:model17
ECHO
set model=BigLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1
goto end
:end
pause