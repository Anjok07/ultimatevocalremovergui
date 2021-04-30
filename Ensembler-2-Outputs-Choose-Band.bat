@echo off

cls
:start
ECHO.
ECHO 1. Ensembler Minimum - 1band_sr44100_hl512
ECHO 2. Ensembler Minimum - 1band_sr32000_hl512
ECHO 3. Ensembler Minimum - 2band_32000
ECHO 4. Ensembler Minimum - 3band_44100
ECHO 5. Ensembler Minimum - 3band_44100_mid
ECHO 6. Ensembler Minimum - 4band_44100
ECHO 7. Ensembler Minimum - 2band_44100_lofi
ECHO 8. Ensembler Maxmimum - 1band_sr44100_hl512
ECHO 9. Ensembler Maxmimum - 1band_sr32000_hl512
ECHO a. Ensembler Maxmimum - 2band_32000
ECHO b. Ensembler Maxmimum - 3band_44100
ECHO c. Ensembler Maxmimum - 3band_44100_mid
ECHO d. Ensembler Maxmimum - 4band_44100
ECHO e. Ensembler Maxmimum - 2band_44100_lofi
set choice=
set /p choice=Type the number associated with the option you would like to run and hit 'Enter': 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='1' goto ensem1
if '%choice%'=='2' goto ensem2
if '%choice%'=='3' goto ensem3
if '%choice%'=='4' goto ensem4
if '%choice%'=='5' goto ensem5
if '%choice%'=='6' goto ensem6
if '%choice%'=='7' goto ensem7
if '%choice%'=='8' goto ensem8
if '%choice%'=='9' goto ensem9
if '%choice%'=='a' goto ensem10
if '%choice%'=='b' goto ensem11
if '%choice%'=='c' goto ensem12
if '%choice%'=='d' goto ensem13
if '%choice%'=='e' goto ensem14
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:ensem1
ECHO
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_min_1band_sr44100_hl512"

goto end
:ensem2
ECHO
set modelparam=1band_sr32000_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_min_1band_sr32000_hl512"
goto end
:ensem3
ECHO
set modelparam=2band_32000
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_min_2band_32000"
goto end
:ensem4
ECHO
set modelparam=3band_44100
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_min_3band_44100"
goto end
:ensem5
ECHO
set modelparam=3band_44100_mid
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_min_3band_44100_mid"
goto end
:ensem6
ECHO
set modelparam=4band_44100
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_min_4band_44100"
goto end
:ensem7
ECHO
set modelparam=2band_44100_lofi
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_2band_44100_lofi"
goto end
:ensem8
ECHO
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_1band_sr44100_hl512"
goto end
:ensem9
ECHO
set modelparam=1band_sr32000_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_1band_sr32000_hl512"
goto end
:ensem10
ECHO
set modelparam=2band_32000
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_2band_32000"
goto end
:ensem11
ECHO
set modelparam=3band_44100
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_3band_44100"
goto end
:ensem12
ECHO
set modelparam=3band_44100_mid
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_3band_44100_mid"
goto end
:ensem13
ECHO
set modelparam=4band_44100
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_4band_44100"
goto end
:ensem14
ECHO
set modelparam=2band_44100_lofi
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json %1 %2 -o ensembled/"%~n1_ensamb_max_2band_44100_lofi"
goto end
:end
pause