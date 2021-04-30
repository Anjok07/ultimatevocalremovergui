@echo off

cls
:start
ECHO =======================================
ECHO Multi-Model Processer and Ensembler
ECHO =======================================
ECHO.
ECHO 1. Ensemble 4Band Model Outputs - 44100Hz
ECHO 2. Ensemble 3Band Model Outputs - 44100Hz
ECHO 3. Ensemble 2Band Model Outputs - 32000Hz
ECHO 4. Exit
set choice=
set /p choice=Type the number associated with your choice and hit 'Enter': 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='1' goto model1
if '%choice%'=='2' goto model2
if '%choice%'=='3' goto model3
if '%choice%'=='4' goto end
if '%choice%'=='' goto end
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:model1
ECHO MODEL MGM-v5-4Band-44100-BETA1
set model=MGM-v5-4Band-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -t -i %1

ECHO MODEL MGM-v5-4Band-44100-BETA2
set model=MGM-v5-4Band-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -P models\%model%.pth -t -i %1

ECHO MODEL HighPrecison_4band_1
set model=HighPrecison_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -t -i %1

ECHO MODEL HighPrecison_4band_2
set model=HighPrecison_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 123821KB -P models\%model%.pth -t -i %1

ECHO MODEL BigLayer_4band_1
set model=BigLayer_4band_1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1

ECHO MODEL BigLayer_4band_2
set model=BigLayer_4band_2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1

ECHO MODEL BigLayer_4band_3
set model=BigLayer_4band_3
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\4band_44100.json -w 352 -n 129605KB -P models\%model%.pth -t -i %1

ECHO Ensembling Instruments...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-4Band-44100-BETA1_Instruments.wav separated\%~n1_MGM-v5-4Band-44100-BETA2_Instruments.wav -o ensembled/temp/"1E2E_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json separated\%~n1_HighPrecison_4band_1_Instruments.wav separated\%~n1_HighPrecison_4band_2_Instruments.wav -o ensembled/temp/"3E4E_ensam2"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json separated\%~n1_BigLayer_4band_1_Instruments.wav separated\%~n1_BigLayer_4band_2_Instruments.wav -o ensembled/temp/"5E6E_ensam3"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/temp/"1E2E_ensam1"_v.wav ensembled/temp/"3E4E_ensam2"_v.wav -o ensembled/temp/"1E2E3E4E_ensam4"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json ensembled/temp/"5E6E_ensam3"_v.wav ensembled/temp/"1E2E3E4E_ensam4"_v.wav -o ensembled/temp/"A6_ensam5"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json separated\%~n1_BigLayer_4band_3_Instruments.wav ensembled/temp/"A6_ensam5"_v.wav -o ensembled/"%~n1_4BAND_Ensembled_Instrumental"

del ensembled\temp\"1E2E_ensam1"_v.wav
del ensembled\temp\"3E4E_ensam2"_v.wav
del ensembled\temp\"5E6E_ensam3"_v.wav
del ensembled\temp\"1E2E3E4E_ensam4"_v.wav
del ensembled\temp\"A6_ensam5"_v.wav

ECHO Ensembling Vocals...

set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-4Band-44100-BETA1_Vocals.wav separated\%~n1_MGM-v5-4Band-44100-BETA2_Vocals.wav -o ensembled/temp/"1EV2EV_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json separated\%~n1_HighPrecison_4band_1_Vocals.wav separated\%~n1_HighPrecison_4band_2_Vocals.wav -o ensembled/temp/"3EV4EV_ensam2"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json separated\%~n1_BigLayer_4band_1_Vocals.wav separated\%~n1_BigLayer_4band_2_Vocals.wav -o ensembled/temp/"5EV6EV_ensam3"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/temp/"1EV2EV_ensam1"_v.wav ensembled/temp/"3EV4EV_ensam2"_v.wav -o ensembled/temp/"1EV2EV3EV4EV_ensam4"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json ensembled/temp/"5EV6EV_ensam3"_v.wav ensembled/temp/"1EV2EV3EV4EV_ensam4"_v.wav -o ensembled/temp/"A6V_ensam5"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json separated\%~n1_BigLayer_4band_3_Vocals.wav ensembled/temp/"A6V_ensam5"_v.wav -o ensembled/"%~n1_4BAND_Ensembled_Vocals"

del ensembled\temp\"1EV2EV_ensam1"_v.wav
del ensembled\temp\"3EV4EV_ensam2"_v.wav
del ensembled\temp\"5EV6EV_ensam3"_v.wav
del ensembled\temp\"1EV2EV3EV4EV_ensam4"_v.wav
del ensembled\temp\"A6V_ensam5"_v.wav

set choice=
set /p choice=Delete Individual Model Outputs? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto delete
if '%choice%'=='y' goto delete
if '%choice%'=='N' goto keep
if '%choice%'=='n' goto keep
if '%choice%'=='' goto keep
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:delete
del separated\%~n1_MGM-v5-4Band-44100-BETA1_Instruments.wav
del separated\%~n1_MGM-v5-4Band-44100-BETA2_Instruments.wav
del separated\%~n1_HighPrecison_4band_1_Instruments.wav
del separated\%~n1_HighPrecison_4band_2_Instruments.wav
del separated\%~n1_BigLayer_4band_1_Instruments.wav
del separated\%~n1_BigLayer_4band_2_Instruments.wav
del separated\%~n1_BigLayer_4band_3_Instruments.wav
del separated\%~n1_MGM-v5-4Band-44100-BETA1_Vocals.wav
del separated\%~n1_MGM-v5-4Band-44100-BETA2_Vocals.wav
del separated\%~n1_HighPrecison_4band_1_Vocals.wav
del separated\%~n1_HighPrecison_4band_2_Vocals.wav
del separated\%~n1_BigLayer_4band_1_Vocals.wav
del separated\%~n1_BigLayer_4band_2_Vocals.wav
del separated\%~n1_BigLayer_4band_3_Vocals.wav
ECHO Complete!
goto start
:keep
ECHO Complete!
goto end
:model2
ECHO MODEL MGM-v5-MIDSIDE-44100-BETA1
set model=MGM-v5-MIDSIDE-44100-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -P models\%model%.pth -t -i %1

ECHO MODEL MGM-v5-MIDSIDE-44100-BETA2
set model=MGM-v5-MIDSIDE-44100-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -P models\%model%.pth -t -i %1

ECHO MODEL MGM-v5-3Band-44100-BETA
set model=MGM-v5-3Band-44100-BETA
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\3band_44100.json -w 352 -P models\%model%.pth -t -i %1

ECHO Ensembling Instruments...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-MIDSIDE-44100-BETA1_Instruments.wav separated\%~n1_MGM-v5-MIDSIDE-44100-BETA2_Instruments.wav -o ensembled/temp/"1E2E_ensam1"

python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-3Band-44100-BETA_Instruments.wav ensembled/temp/"1E2E_ensam1"_v.wav -o ensembled/"%~n1_3BAND_Ensembled_Instrumental"

del ensembled\temp\"1E2E_ensam1"_v.wav

ECHO Ensembling Vocals...

set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-MIDSIDE-44100-BETA1_Vocals.wav separated\%~n1_MGM-v5-MIDSIDE-44100-BETA2_Vocals.wav -o ensembled/temp/"1EV2EV_ensam1"

python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-3Band-44100-BETA_Vocals.wav ensembled/temp/"1EV2EV_ensam1"_v.wav -o ensembled/"%~n1_3BAND_Ensembled_Vocals"

del ensembled\temp\"1EV2EV_ensam1"_v.wav

set choice=
set /p choice=Delete Individual Model Outputs? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto delete
if '%choice%'=='y' goto delete
if '%choice%'=='N' goto keep
if '%choice%'=='n' goto keep
if '%choice%'=='' goto keep
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:delete
del separated\%~n1_MGM-v5-MIDSIDE-44100-BETA1_Instruments.wav
del separated\%~n1_MGM-v5-MIDSIDE-44100-BETA2_Instruments.wav
del separated\%~n1_MGM-v5-3Band-44100-BETA_Instruments.wav
del separated\%~n1_MGM-v5-MIDSIDE-44100-BETA1_Vocals.wav
del separated\%~n1_MGM-v5-MIDSIDE-44100-BETA2_Vocals.wav
del separated\%~n1_MGM-v5-3Band-44100-BETA_Vocals.wav
ECHO Complete!
goto start
:keep
ECHO Complete!
goto end
:model3
ECHO MODEL MGM-v5-2Band-32000-BETA1
set model=MGM-v5-2Band-32000-BETA1
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1

ECHO MODEL MGM-v5-2Band-32000-BETA2
set model=MGM-v5-2Band-32000-BETA2
cd /d %~dp0
 
python inference.py -g 0 -m modelparams\2band_32000.json -w 352 -P models\%model%.pth -t -i %1

ECHO Ensembling Instruments...
set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a min_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-2Band-32000-BETA1_Instruments.wav separated\%~n1_MGM-v5-2Band-32000-BETA2_Instruments.wav -o ensembled/"%~n1_2BAND_Ensembled_Instrumental"

ECHO Ensembling Vocals...

set modelparam=1band_sr44100_hl512
cd /d %~dp0
 
python lib/spec_utils.py -a max_mag -m modelparams\%modelparam%.json separated\%~n1_MGM-v5-2Band-32000-BETA1_Vocals.wav separated\%~n1_MGM-v5-2Band-32000-BETA2_Vocals.wav -o ensembled/"%~n1_2BAND_Ensembled_Vocals"

set choice=
set /p choice=Delete Individual Model Outputs? [Y/N]: 
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='Y' goto delete
if '%choice%'=='y' goto delete
if '%choice%'=='N' goto keep
if '%choice%'=='n' goto keep
if '%choice%'=='' goto keep
ECHO "%choice%" is not valid, try again
ECHO.
goto start
:delete
del separated\%~n1_MGM-v5-2Band-32000-BETA1_Instruments.wav
del separated\%~n1_MGM-v5-2Band-32000-BETA2_Instruments.wav
del separated\%~n1_MGM-v5-2Band-32000-BETA1_Vocals.wav
del separated\%~n1_MGM-v5-2Band-32000-BETA2_Vocals.wav
ECHO Complete!
goto start
:keep
ECHO Complete!
goto end
:end