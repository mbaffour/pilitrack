@echo off
title Install pilitrack
echo ============================================================
echo   Installing pilitrack and everything it needs.
echo   First run downloads a lot and can take several minutes.
echo ============================================================
echo.
cd /d "%~dp0"
REM Find and activate a conda / miniconda base so pip installs into it.
for %%P in (
  "%USERPROFILE%\Miniconda3" "%USERPROFILE%\Anaconda3"
  "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3"
  "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\Continuum\miniconda3"
) do (
  if exist "%%~P\Scripts\activate.bat" ( call "%%~P\Scripts\activate.bat" & goto :install )
)
:install
python -m pip install --upgrade pip
python -m pip install -e "pilitrack_pkg[all]"
if errorlevel 1 (
  echo.
  echo   Install failed.
  echo   If you don't have conda yet, install Miniconda first:
  echo       https://docs.conda.io/en/latest/miniconda.html
  echo   then double-click this file again.
  echo.
  pause
  exit /b 1
)
echo.
echo   All set! Double-click "Start pilitrack (browser)" to launch.
echo.
pause
