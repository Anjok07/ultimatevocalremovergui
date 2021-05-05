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
ECHO a. MGM-v5-2Band-32000-BETA1
ECHO b. MGM-v5-2Band-32000-BETA2
ECHO c. MGM-v5-3Band-44100-BETA
ECHO d. MGM-v5-4Band-44100-BETA1
ECHO e. MGM-v5-4Band-44100-BETA2
ECHO f. MGM-v5-KAROKEE-32000-BETA1
ECHO g. MGM-v5-KAROKEE-32000-BETA2-AGR
ECHO h. MGM-v5-MIDSIDE-44100-BETA1
ECHO i. MGM-v5-MIDSIDE-44100-BETA2
ECHO j. MGM-v5-Vocal_2Band-32000-BETA1
ECHO k. MGM-v5-Vocal_2Band-32000-BETA2
ECHO l. LOFI_2band_iter5_2
ECHO m. LOFI_2band_iter5_2
ECHO n. HighPrecison_4band_1
ECHO o. HighPrecison_4band_2
ECHO p. NewLayer_4band_1.pth
ECHO q. NewLayer_4band_2.pth
ECHO r. NewLayer_4band_3.pth
ECHO s. Ensemble 4Band Models (7 Models) - Final Outputs Save to Ensembled Folder!
ECHO t. Ensemble All Models (12 Models) - Final Outputs Save to Ensembled Folder!
set choice=
set /p choice=Type the letter associated with the option you choose and hit 'Enter': 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='a' goto model1
if '%choice%'=='b' goto model2
if '%choice%'=='c' goto model3
if '%choice%'=='d' goto model4
if '%choice%'=='e' goto model5
if '%choice%'=='f' goto model6
if '%choice%'=='g' goto model7
if '%choice%'=='h' goto model8
if '%choice%'=='i' goto model9
if '%choice%'=='j' goto model10
if '%choice%'=='k' goto model11
if '%choice%'=='l' goto model12
if '%choice%'=='m' goto model13
if '%choice%'=='n' goto model14
if '%choice%'=='o' goto model15
if '%choice%'=='p' goto model16
if '%choice%'=='q' goto model17
if '%choice%'=='r' goto model18
if '%choice%'=='s' goto model19
if '%choice%'=='t' goto model20
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
:default

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
:default

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
:default

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
:default

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
:default

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
:default

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
ECHO Running Model NewLayer_4band_1
set model=NewLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -t -i %1
goto start
:default
ECHO Running Model NewLayer_4band_1
set model=NewLayer_4band_1
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

ECHO Running Model NewLayer_4band_2
set model=NewLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running Model NewLayer_4band_2
set model=NewLayer_4band_2
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

ECHO Running NewLayer_4band_3
set model=NewLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -t -i %1
goto end
:default

ECHO Running NewLayer_4band_3
set model=NewLayer_4band_3
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
 
python 4Band_ens_inference.py -g 0 -D -w 352 -D -s -t -i %1
goto end
:default

ECHO Ensemble All 4Band Models
cd /d %~dp0
 
python 4Band_ens_inference.py -g 0 -w 352 -D -s -t -i %1
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
:default

ECHO Ensemble All 44100-Models
cd /d %~dp0
 
python allmodels_ens_inference.py -g 0 -w 352 -t -i %1
pause
:notta
ECHO =======================================
ECHO           Choose Your Model
ECHO =======================================
ECHO.
ECHO a. MGM-v5-2Band-32000-BETA1
ECHO b. MGM-v5-2Band-32000-BETA2
ECHO c. MGM-v5-3Band-44100-BETA
ECHO d. MGM-v5-4Band-44100-BETA1
ECHO e. MGM-v5-4Band-44100-BETA2
ECHO f. MGM-v5-KAROKEE-32000-BETA1
ECHO g. MGM-v5-KAROKEE-32000-BETA2-AGR
ECHO h. MGM-v5-MIDSIDE-44100-BETA1
ECHO i. MGM-v5-MIDSIDE-44100-BETA2
ECHO j. MGM-v5-Vocal_2Band-32000-BETA1
ECHO k. MGM-v5-Vocal_2Band-32000-BETA2
ECHO l. LOFI_2band_iter5_2
ECHO m. LOFI_2band_iter5_2
ECHO n. HighPrecison_4band_1
ECHO o. HighPrecison_4band_2
ECHO p. NewLayer_4band_1
ECHO q. NewLayer_4band_2
ECHO r. NewLayer_4band_3
ECHO s. Ensemble 4Band Models (7 Models) - Final Outputs Save to Ensembled Folder!
ECHO t. Ensemble All Models (12 Models) - Final Outputs Save to Ensembled Folder!
set choice=
set /p choice=Type the letter associated with the option you choose and hit 'Enter': 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='a' goto model1
if '%choice%'=='b' goto model2
if '%choice%'=='c' goto model3
if '%choice%'=='d' goto model4
if '%choice%'=='e' goto model5
if '%choice%'=='f' goto model6
if '%choice%'=='g' goto model7
if '%choice%'=='h' goto model8
if '%choice%'=='i' goto model9
if '%choice%'=='j' goto model10
if '%choice%'=='k' goto model11
if '%choice%'=='l' goto model12
if '%choice%'=='m' goto model13
if '%choice%'=='n' goto model14
if '%choice%'=='o' goto model15
if '%choice%'=='p' goto model16
if '%choice%'=='q' goto model17
if '%choice%'=='r' goto model18
if '%choice%'=='s' goto model19
if '%choice%'=='t' goto model20
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
:default

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
:default

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
:default

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
:default

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
:default

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
:default

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
ECHO Running Model NewLayer_4band_1
set model=NewLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -i %1
goto start
:default
ECHO Running Model NewLayer_4band_1
set model=NewLayer_4band_1
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

ECHO Running Model NewLayer_4band_2
set model=NewLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -i %1
goto end
:default

ECHO Running Model NewLayer_4band_2
set model=NewLayer_4band_2
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

ECHO Running NewLayer_4band_3
set model=NewLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -D -n 129605KB -P models\%model%.pth -i %1
goto end
:default

ECHO Running NewLayer_4band_3
set model=NewLayer_4band_3
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
:default

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
:default

ECHO Ensemble All 44100-Models
cd /d %~dp0
 
python allmodels_ens_inference.py -g 0 -w 352 -i %1
goto end
:end
pause