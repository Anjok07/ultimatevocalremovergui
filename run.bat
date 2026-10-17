@echo off
REM Launch the UVR web service natively (Windows).
REM First run sets up a venv + installs deps + builds the UI; later runs just start.
REM Override a setting by setting it before running, e.g.  set UVR_PORT=9000 && run.bat
setlocal enableextensions
cd /d "%~dp0"

REM ---------------------------------------------------------------- configuration
if "%UVR_HOST%"=="" set "UVR_HOST=127.0.0.1"
if "%UVR_PORT%"=="" set "UVR_PORT=8000"
if "%UVR_DATA_DIR%"=="" set "UVR_DATA_DIR=%cd%\data"
if "%UVR_OPEN_BROWSER%"=="" set "UVR_OPEN_BROWSER=1"
if "%UVR_NUM_THREADS%"=="" set "UVR_NUM_THREADS=2"
if "%OMP_NUM_THREADS%"=="" set "OMP_NUM_THREADS=%UVR_NUM_THREADS%"
if "%MKL_NUM_THREADS%"=="" set "MKL_NUM_THREADS=%UVR_NUM_THREADS%"
if "%UVR_USE_GPU%"=="" set "UVR_USE_GPU=auto"
set "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True"

set "VENV=.venv"

REM ------------------------------------------------------------- first-run set-up
if not exist "%VENV%\Scripts\python.exe" (
  echo ^>^> Creating virtualenv and installing dependencies ^(first run^)...
  python -m venv "%VENV%" || goto :error
  "%VENV%\Scripts\python" -m pip install --upgrade pip wheel || goto :error
  REM Windows PyPI torch wheels are CPU builds.
  "%VENV%\Scripts\pip" install torch==2.2.2 torchvision==0.17.2 || goto :error
  "%VENV%\Scripts\pip" install -r requirements.txt -r requirements-api.txt || goto :error
)

REM --------------------------------------------------------------- build the UI
if not exist "web\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo !! web\dist not found and npm is not installed - the UI won't be served.
    echo    ^(the API still works; install Node + run "cd web ^&^& npm ci ^&^& npm run build"^)
  ) else (
    echo ^>^> Building the web UI...
    pushd web
    call npm ci || goto :error
    call npm run build || goto :error
    popd
  )
)

REM ---------------------------------------------------------------- data folders
if not exist "%UVR_DATA_DIR%\inputs"  mkdir "%UVR_DATA_DIR%\inputs"
if not exist "%UVR_DATA_DIR%\jobs"    mkdir "%UVR_DATA_DIR%\jobs"
if not exist "%UVR_DATA_DIR%\uploads" mkdir "%UVR_DATA_DIR%\uploads"

REM -------------------------------------------------------------------------- run
echo ^>^> UVR web service -^> http://%UVR_HOST%:%UVR_PORT%   (data: %UVR_DATA_DIR%)
"%VENV%\Scripts\python" run_web.py
goto :eof

:error
echo.
echo Setup failed. On Windows, building diffq needs "Microsoft C++ Build Tools"
echo (Desktop development with C++). See API_README.md.
exit /b 1
