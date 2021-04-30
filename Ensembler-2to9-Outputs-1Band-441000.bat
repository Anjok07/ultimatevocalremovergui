@echo off

cls
:start
ECHO.
ECHO 2. Enter if two audio files were dropped.
ECHO 3. Enter if three audio files were dropped.
ECHO 4. Enter if four audio files were dropped.
ECHO 5. Enter if five audio files were dropped.
ECHO 6. Enter if six audio files were dropped.
ECHO 7. Enter if seven audio files were dropped.
ECHO 8. Enter if eight audio files were dropped.
ECHO 9. Enter if nine audio files were dropped.
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
ECHO
set modelparam=1band_sr44100_hl512
cd /d %~dp0

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_min_1band_sr44100_hl512"
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
ECHO
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %3 %4 -o ensembled/"%~n3_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/"%~n1_ensam1"_v.wav ensembled/"%~n3_ensam2"_v.wav -o ensembled/"%~n1_Final_Ensemb_4"

del ensembled\"%~n1_ensam1"_v.wav
del ensembled\"%~n3_ensam2"_v.wav
goto end
:ensem5
ECHO
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
ECHO
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
ECHO
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
ECHO
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
ECHO
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
pause