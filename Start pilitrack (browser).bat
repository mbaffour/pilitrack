@echo off
title pilitrack (browser)
echo.
echo   Starting pilitrack in your browser...
echo   (a small local server - keep this window open; close it to stop)
echo.
REM Activate a conda/miniconda base so the pilitrack-web command is on PATH.
for %%P in (
  "%USERPROFILE%\Miniconda3" "%USERPROFILE%\Anaconda3"
  "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3"
  "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\Continuum\miniconda3"
) do (
  if exist "%%~P\Scripts\activate.bat" (
    call "%%~P\Scripts\activate.bat"
    goto :run
  )
)
:run
pilitrack-web
if errorlevel 1 (
  echo.
  echo   Could not start automatically. Open your Anaconda/Miniconda Prompt and run:
  echo       pilitrack-web
  echo.
  pause
)
