@echo off

:start
set choice=
set /p choice=Are you processing instrumental outputs? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto inst
if '%choice%'=='y' goto inst
if '%choice%'=='N' goto vocals
if '%choice%'=='n' goto vocals
if '%choice%'=='' goto inst

:inst
ECHO Ensembling Instruments...
set modelparam=1band_sr44100_hl512
cd /d %~dp0

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %* -o "%~n1_Ensembled_Instruments"
ECHO Complete!
goto end
:end
pause
exit
:vocals
ECHO Ensembling Vocals...
set modelparam=1band_sr44100_hl512
cd /d %~dp0

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %* -o "%~n1_Ensembled_Vocals"
ECHO Complete!
goto end
:end
pause
exit
