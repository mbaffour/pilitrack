@echo off
title pilitrack (desktop)
echo.
echo   Opening pilitrack... choose a movie when the window appears.
echo.
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
pilitrack-gui
if errorlevel 1 (
  echo.
  echo   Could not start automatically. Open your Anaconda/Miniconda Prompt and run:
  echo       pilitrack-gui
  echo.
  pause
)
