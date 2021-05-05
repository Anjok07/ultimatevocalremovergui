@echo off

cls
:start
set choice=
set /p choice=Run with TTA? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto tta
if '%choice%'=='y' goto tta
if '%choice%'=='N' goto notta
if '%choice%'=='n' goto notta
if '%choice%'=='' goto notta
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:tta
ECHO =======================================
ECHO           Choose Your Model
ECHO =======================================
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
ECHO d. LOFI_2band_iter5_2
ECHO e. HighPrecison_4band_1
ECHO f. HighPrecison_4band_2
ECHO g. BigLayer_4band_1.pth
ECHO h. BigLayer_4band_2.pth
ECHO i. BigLayer_4band_3.pth
ECHO j. Ensemble 4Band Models (7 Models) - Final Outputs Save to Ensembled Folder!
ECHO k. Ensemble All Models (12 Models) - Final Outputs Save to Ensembled Folder!
set choice=
set /p choice=Type the number or letter associated with the option you choose and hit 'Enter': 
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
if '%choice%'=='i' goto model18
if '%choice%'=='j' goto model19
if '%choice%'=='k' goto model20
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:model1
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep
ECHO Running MGM-v5-2Band-32000-BETA1
set model=MGM-v5-2Band-32000-BETA1
cd /d %~dp0

python inference.py -g 0 -m modelparams\2band_32000.json -D -w 352 -P models\%model%.pth -t -i %1
goto end
:default
ECHO Running MGM-v5-2Band-32000-BETA1
set model=MGM-v5-2Band-32000-BETA1
cd /d %~dp0

python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model2
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-2Band-32000-BETA2
set model=Running Model MGM-v5-2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -D -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running Model MGM-v5-2Band-32000-BETA2
set model=Running Model MGM-v5-2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model3
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-3Band-44100-BETA
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -D -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running Model MGM-v5-3Band-44100-BETA
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model4
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-4Band-44100-BETA1
set model=MGM-v5-4Band-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running Model MGM-v5-4Band-44100-BETA1
set model=MGM-v5-4Band-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model5
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-4Band-44100-BETA2
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -D -w 352 -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running Model MGM-v5-4Band-44100-BETA2
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model6
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-KAROKEE-32000-BETA1
set model=MGM-v5-KAROKEE-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -D -w 352 -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running Model MGM-v5-KAROKEE-32000-BETA1
set model=MGM-v5-KAROKEE-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model7
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-KAROKEE-32000-BETA2-AGR
set model=MGM-v5-KAROKEE-32000-BETA2-AGR
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -D -w 352 -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running Model MGM-v5-KAROKEE-32000-BETA2-AGR
set model=MGM-v5-KAROKEE-32000-BETA2-AGR
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model8
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA1
set model=MGM-v5-MIDSIDE-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -D -w 352 -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA1
set model=MGM-v5-MIDSIDE-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model9
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA2
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -D -w 352 -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA2
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model10
ECHO Running Model MGM-v5-Vocal_2Band-32000-BETA1
set model=MGM-v5-Vocal_2Band-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model11
ECHO Running Model MGM-v5-Vocal_2Band-32000-BETA2
set model=MGM-v5-Vocal_2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1
goto end
:model12
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model LOFI_2band-1_33966KB
set model=LOFI_2band-32000-1_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -D -w 352 -n 33966KB -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model LOFI_2band-1_33966KB
set model=LOFI_2band-32000-1_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -w 352 -n 33966KB -P models\%model%.pth -t -i %1
goto end
:model13
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model LOFI_2band-2_33966KB
set model=LOFI_2band-2_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -D -w 352 -n 33966KB -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model LOFI_2band-2_33966KB
set model=LOFI_2band-2_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -w 352 -n 33966KB -P models\%model%.pth -t -i %1
goto end
:model14
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model HighPrecison_4band_1
set model=HighPrecison_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -D -w 352 -n 123821KB -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model HighPrecison_4band_1
set model=HighPrecison_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -t -i %1
goto end
:model15
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model HighPrecison_4band_2
set model=HighPrecison_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -D -w 352 -n 123821KB -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model HighPrecison_4band_2
set model=HighPrecison_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -t -i %1
goto end
:model16
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep
ECHO Running Model BigLayer_4band_1
set model=BigLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -t -i %1
goto start
:default
ECHO Running Model BigLayer_4band_1
set model=BigLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1
goto end
:model17
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model BigLayer_4band_2
set model=BigLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model BigLayer_4band_2
set model=BigLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1
goto end
:model18
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model 
set model=BigLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -t -i %1
goto end
:regular

ECHO Running Model 
set model=BigLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1
goto end
:model19
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Ensemble All 4Band Models
cd /d %~dp0
 
python 4Band_ens_inference.py -g 0 -D -w 352 -D -t -i %1
goto end
:regular

ECHO Ensemble All 4Band Models
cd /d %~dp0
 
python 4Band_ens_inference.py -g 0 -w 352 -D -t -i %1
goto end
:model20
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Ensemble All 44100-Models
cd /d %~dp0
 
python allmodels_ens_inference.py -g 0 -w 352 -D -t -i %1
goto end
:regular

ECHO Ensemble All 44100-Models
cd /d %~dp0
 
python allmodels_ens_inference.py -g 0 -w 352 -t -i %1
pause
:notta
ECHO =======================================
ECHO           Choose Your Model
ECHO =======================================
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
ECHO d. LOFI_2band_iter5_2
ECHO e. HighPrecison_4band_1
ECHO f. HighPrecison_4band_2
ECHO g. BigLayer_4band_1.pth
ECHO h. BigLayer_4band_2.pth
ECHO i. BigLayer_4band_3.pth
ECHO j. Ensemble 4Band Models (7 Models)
ECHO k. Ensemble All Models (14 Models)
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
if '%choice%'=='i' goto model18
if '%choice%'=='j' goto model19
if '%choice%'=='k' goto model20
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:model1
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep
ECHO Running MGM-v5-2Band-32000-BETA1
set model=MGM-v5-2Band-32000-BETA1
cd /d %~dp0

python inference.py -g 0 -m modelparams\2band_32000.json -D -w 352 -P models\%model%.pth -i %1
goto end
:default
ECHO Running MGM-v5-2Band-32000-BETA1
set model=MGM-v5-2Band-32000-BETA1
cd /d %~dp0

python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -i %1
goto end
:model2
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-2Band-32000-BETA2
set model=Running Model MGM-v5-2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -D -P models\%model%.pth -i %1
goto end
:default

ECHO Running Model MGM-v5-2Band-32000-BETA2
set model=Running Model MGM-v5-2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -i %1
goto end
:model3
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-3Band-44100-BETA
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -D -P models\%model%.pth -i %1
goto end
:default

ECHO Running Model MGM-v5-3Band-44100-BETA
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -P models\%model%.pth -i %1
goto end
:model4
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-4Band-44100-BETA1
set model=MGM-v5-4Band-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -P models\%model%.pth -i %1
goto end
:default

ECHO Running Model MGM-v5-4Band-44100-BETA1
set model=MGM-v5-4Band-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -i %1
goto end
:model5
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-4Band-44100-BETA2
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -D -w 352 -P models\%model%.pth -i %1
goto end
:default

ECHO Running Model MGM-v5-4Band-44100-BETA2
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -i %1
goto end
:model6
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-KAROKEE-32000-BETA1
set model=MGM-v5-KAROKEE-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -D -w 352 -P models\%model%.pth -i %1
goto end
:default

ECHO Running Model MGM-v5-KAROKEE-32000-BETA1
set model=MGM-v5-KAROKEE-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -i %1
goto end
:model7
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-KAROKEE-32000-BETA2-AGR
set model=MGM-v5-KAROKEE-32000-BETA2-AGR
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -D -w 352 -P models\%model%.pth -i %1
goto end
:default

ECHO Running Model MGM-v5-KAROKEE-32000-BETA2-AGR
set model=MGM-v5-KAROKEE-32000-BETA2-AGR
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -i %1
goto end
:model8
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA1
set model=MGM-v5-MIDSIDE-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -D -w 352 -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA1
set model=MGM-v5-MIDSIDE-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -w 352 -P models\%model%.pth -i %1
goto end
:model9
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA2
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -D -w 352 -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model MGM-v5-MIDSIDE-44100-BETA2
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100_mid.json -w 352 -P models\%model%.pth -i %1
goto end
:model10
ECHO Running Model MGM-v5-Vocal_2Band-32000-BETA1
set model=MGM-v5-Vocal_2Band-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -i %1
goto end
:model11
ECHO Running Model MGM-v5-Vocal_2Band-32000-BETA2
set model=MGM-v5-Vocal_2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -i %1
goto end
:model12
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model LOFI_2band-1_33966KB
set model=LOFI_2band-32000-1_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -D -w 352 -n 33966KB -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model LOFI_2band-1_33966KB
set model=LOFI_2band-32000-1_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -w 352 -n 33966KB -P models\%model%.pth -i %1
goto end
:model13
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model LOFI_2band-2_33966KB
set model=LOFI_2band-2_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -D -w 352 -n 33966KB -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model LOFI_2band-2_33966KB
set model=LOFI_2band-2_33966KB
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_44100_lofi.json -w 352 -n 33966KB -P models\%model%.pth -i %1
goto end
:model14
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model HighPrecison_4band_1
set model=HighPrecison_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -D -w 352 -n 123821KB -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model HighPrecison_4band_1
set model=HighPrecison_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -i %1
goto end
:model15
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model HighPrecison_4band_2
set model=HighPrecison_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -D -w 352 -n 123821KB -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model HighPrecison_4band_2
set model=HighPrecison_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -i %1
goto end
:model16
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep
ECHO Running Model BigLayer_4band_1
set model=BigLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -i %1
goto start
:default
ECHO Running Model BigLayer_4band_1
set model=BigLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -i %1
goto end
:model17
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model BigLayer_4band_2
set model=BigLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model BigLayer_4band_2
set model=BigLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -i %1
goto end
:model18
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Running Model 
set model=BigLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -i %1
goto end
:regular

ECHO Running Model 
set model=BigLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -i %1
goto end
:model19
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Ensemble All 4Band Models
cd /d %~dp0
 
python 4Band_ens_inference.py -g 0 -D -w 352 -D -i %1
goto end
:regular

ECHO Ensemble All 4Band Models
cd /d %~dp0
 
python 4Band_ens_inference.py -g 0 -w 352 -D -i %1
goto end
:model20
set choice=
set /p choice=Include Deep Vocal Extraction Instrumental? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto deep
if '%choice%'=='y' goto deep
if '%choice%'=='N' goto default
if '%choice%'=='n' goto default
if '%choice%'=='' goto default
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:deep

ECHO Ensemble All 44100-Models
cd /d %~dp0
 
python allmodels_ens_inference.py -g 0 -w 352 -D -i %1
goto end
:regular

ECHO Ensemble All 44100-Models
cd /d %~dp0
 
python allmodels_ens_inference.py -g 0 -w 352 -i %1
:end
pause