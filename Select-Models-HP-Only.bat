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
ECHO a. HP2-4BAND-3090_4band_1
ECHO b. HP2-4BAND-3090_4band_1
ECHO c. HP_4BAND_3090
ECHO d. Vocal_HP_4BAND_3090
ECHO e. Vocal_HP_4BAND_3090_AGG
ECHO f. Ensemble All HP Instrumental Models (5 Models) - Final Outputs Save to Ensembled Folder!
ECHO g. Ensemble HP2 Models Only (3 Models) - Final Outputs Save to Ensembled Folder!
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
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:model1
ECHO Running HP2-4BAND-3090_4band_1
set model=HP2-4BAND-3090_4band_1
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 537238KB -w 352 -P models\%model%.pth -t -i %1
goto end
:model2
ECHO Running HP2-4BAND-3090_4band_2
set model=HP2-4BAND-3090_4band_3
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 537238KB -w 352 -P models\%model%.pth -t -i %1
goto end
:model3
ECHO Running HP_4BAND_3090
set model=HP_4BAND_3090
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 123821KB -w 352 -P models\%model%.pth -t -i %1
goto end
:model4
ECHO Running Vocal_HP_4BAND_3090
set model=Vocal_HP_4BAND_3090
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 123821KB -w 352 -P models\%model%.pth -t -vm -i %1
goto end
:model5
ECHO Running Vocal_HP_4BAND_3090_AGG
set model=Vocal_HP_4BAND_3090_AGG
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 123821KB -w 352 -P models\%model%.pth -t -vm -i %1
goto end
:model6
ECHO Running and Ensembling All HP Instrumental Models
cd /d %~dp0

python ensemble_inference.py -g 0 -w 352 -P HighPrecison_4band_1 HighPrecison_4band_2 HP_4BAND_3090 HP2-4BAND-3090_4band_1 HP2-4BAND-3090_4band_2 HP2-4BAND-3090_4band_3 -s -t -i %1
goto end
:model7
ECHO Running and Ensembling HP2 Instrumental Models
cd /d %~dp0

python ensemble_inference.py -g 0 -w 352 -P HP2-4BAND-3090_4band_1 HP2-4BAND-3090_4band_2 HP2-4BAND-3090_4band_3 -s -t -i %1
goto end
:end
ECHO --------------------------------
pause
exit
:notta
ECHO =======================================
ECHO           Choose Your Model
ECHO =======================================
ECHO.
ECHO a. HP2-4BAND-3090_4band_1
ECHO b. HP2-4BAND-3090_4band_2
ECHO c. HP_4BAND_3090
ECHO d. Vocal_HP_4BAND_3090
ECHO e. Vocal_HP_4BAND_3090_AGG
ECHO f. Ensemble All HP Instrumental Models (5 Models) - Final Outputs Save to Ensembled Folder!
ECHO g. Ensemble HP2 Models Only (3 Models) - Final Outputs Save to Ensembled Folder!
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
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:model1
ECHO Running HP2-4BAND-3090_4band_1
set model=HP2-4BAND-3090_4band_1
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 537238KB -w 352 -P models\%model%.pth -i %1
goto end
:model2
ECHO Running HP2-4BAND-3090_4band_2
set model=HP2-4BAND-3090_4band_2
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 537238KB -w 352 -P models\%model%.pth -i %1
goto end
:model3
ECHO Running HP_4BAND_3090
set model=HP_4BAND_3090
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 123821KB -w 352 -P models\%model%.pth -i %1
goto end
:model4
ECHO Running Vocal_HP_4BAND_3090
set model=Vocal_HP_4BAND_3090
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 123821KB -w 352 -P models\%model%.pth -vm -i %1
goto end
:model5
ECHO Running Vocal_HP_4BAND_3090_AGG
set model=Vocal_HP_4BAND_3090_AGG
cd /d %~dp0

python inference.py -g 0 -m modelparams\4band_44100.json -n 123821KB -w 352 -P models\%model%.pth -vm -i %1
goto end
:model6
ECHO Running and Ensembling All HP Instrumental Models
cd /d %~dp0

python ensemble_inference.py -g 0 -w 352 -P HighPrecison_4band_1 HighPrecison_4band_2 HP_4BAND_3090 HP2-4BAND-3090_4band_1 HP2-4BAND-3090_4band_2 HP2-4BAND-3090_4band_3 -s -i %1
goto end
:model7
ECHO Running and Ensembling HP2 Instrumental Models
cd /d %~dp0

python ensemble_inference.py -g 0 -w 352 -P HP2-4BAND-3090_4band_1 HP2-4BAND-3090_4band_2 HP2-4BAND-3090_4band_3 -s -i %1
goto end
:end
ECHO --------------------------------
pause
exit