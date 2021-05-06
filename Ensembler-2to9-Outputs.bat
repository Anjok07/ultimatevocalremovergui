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
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:inst
ECHO =======================================
ECHO     Instrumental Output Ensembler
ECHO =======================================
ECHO.
set choice=
set /p choice=Please enter the number of files you are processing: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='2' goto ensem2
if '%choice%'=='3' goto ensem3
if '%choice%'=='4' goto ensem4
if '%choice%'=='5' goto ensem5
if '%choice%'=='6' goto ensem6
if '%choice%'=='7' goto ensem7
if '%choice%'=='8' goto ensem8
if '%choice%'=='9' goto ensem9
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:ensem2
ECHO Ensembling instrumental outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_Final_Ensemb_2"
goto end
:ensem3
ECHO
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 ensembled/"%~n1_ensam1"_v.wav -o ensembled/"%~n1_Final_Ensemb_3"

del ensembled\"%~n1_ensam1"_v.wav
goto end
:ensem4
ECHO Ensembling instrumental outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_Final_Ensemb_4"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
goto end
:ensem5
ECHO Ensembling instrumental outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam3"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %5 ensembled/"%~n1_ensam3"_v.wav -o ensembled/"%~n1_Final_Ensemb_5"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n1_ensam3"_v.wav
goto end
:ensem6
ECHO Ensembling instrumental outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam4"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n1_ensam4"_v.wav -o ensembled/"%~n1_Final_Ensemb_6"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n1_ensam4"_v.wav
goto end
:ensem7
ECHO Ensembling instrumental outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam4"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n1_ensam4"_v.wav -o ensembled/"%~n1_ensam5"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %7 ensembled/"%~n1_ensam5"_v.wav -o ensembled/"%~n1_Final_Ensemb_7"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n1_ensam4"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n1_ensam5"_v.wav
goto end
:ensem8
ECHO Ensembling instrumental outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %7 %8 -o ensembled/"%~n7_ensam4"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam12"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n7_ensam4"_v.wav -o ensembled/"%~n1_ensam34"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam12"_v.wav ensembled/"%~n1_ensam34"_v.wav -o ensembled/"%~n1_Final_Ensemb_8"
 
del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n7_ensam4"_v.wav
del ensembled\"%~n1_ensam12"_v.wav
del ensembled\"%~n1_ensam34"_v.wav
goto end
:ensem9
ECHO Ensembling instrumental outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %7 %8 -o ensembled/"%~n7_ensam4"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam12"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n7_ensam4"_v.wav -o ensembled/"%~n1_ensam34"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam12"_v.wav ensembled/"%~n1_ensam34"_v.wav -o ensembled/"%~n1_ensam1234"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %9 ensembled/"%~n1_ensam1234"_v.wav -o ensembled/"%~n1_Final_Ensemb_9"
 
del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n7_ensam4"_v.wav
del ensembled\"%~n1_ensam12"_v.wav
del ensembled\"%~n1_ensam34"_v.wav
del ensembled\"%~n1_ensam1234"_v.wav
goto end
:end
ECHO Complete!
pause
exit
:vocals
ECHO =======================================
ECHO        Vocal Output Ensembler
ECHO =======================================
ECHO.
set choice=
set /p choice=Please enter the number of files you are processing: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='2' goto ensem2
if '%choice%'=='3' goto ensem3
if '%choice%'=='4' goto ensem4
if '%choice%'=='5' goto ensem5
if '%choice%'=='6' goto ensem6
if '%choice%'=='7' goto ensem7
if '%choice%'=='8' goto ensem8
if '%choice%'=='9' goto ensem9
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:ensem2
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_Final_Ensemb_2"
goto end
:ensem3
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %3 ensembled/"%~n1_ensam1"_v.wav -o ensembled/"%~n1_Final_Ensemb_3"

del ensembled\"%~n1_ensam1"_v.wav
goto end
:ensem4
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_Final_Ensemb_4"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
goto end
:ensem5
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam3"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %5 ensembled/"%~n1_ensam3"_v.wav -o ensembled/"%~n1_Final_Ensemb_5"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n1_ensam3"_v.wav
goto end
:ensem6
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam4"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n1_ensam4"_v.wav -o ensembled/"%~n1_Final_Ensemb_6"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n1_ensam4"_v.wav
goto end
:ensem7
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam4"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n1_ensam4"_v.wav -o ensembled/"%~n1_ensam5"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %7 ensembled/"%~n1_ensam5"_v.wav -o ensembled/"%~n1_Final_Ensemb_7"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n1_ensam4"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n1_ensam5"_v.wav
goto end
:ensem8
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %7 %8 -o ensembled/"%~n7_ensam4"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam12"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n7_ensam4"_v.wav -o ensembled/"%~n1_ensam34"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam12"_v.wav ensembled/"%~n1_ensam34"_v.wav -o ensembled/"%~n1_Final_Ensemb_8"
 
del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n7_ensam4"_v.wav
del ensembled\"%~n1_ensam12"_v.wav
del ensembled\"%~n1_ensam34"_v.wav
goto end
:ensem9
ECHO Ensembling vocal outputs...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %5 %6 -o ensembled/"%~n5_ensam3"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %7 %8 -o ensembled/"%~n7_ensam4"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_ensam12"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n5_ensam3"_v.wav ensembled/"%~n7_ensam4"_v.wav -o ensembled/"%~n1_ensam34"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam12"_v.wav ensembled/"%~n1_ensam34"_v.wav -o ensembled/"%~n1_ensam1234"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %9 ensembled/"%~n1_ensam1234"_v.wav -o ensembled/"%~n1_Final_Ensemb_9"
 
del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
del ensembled\"%~n5_ensam3"_v.wav
del ensembled\"%~n7_ensam4"_v.wav
del ensembled\"%~n1_ensam12"_v.wav
del ensembled\"%~n1_ensam34"_v.wav
del ensembled\"%~n1_ensam1234"_v.wav
goto end
:end
ECHO Complete!
pause
exit