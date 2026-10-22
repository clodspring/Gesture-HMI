@echo off
setlocal
cd /d "%~dp0"
echo === Gesture HMI - Local Windows EXE Builder ===
echo This builder does not depend on your Conda environment.
echo Internet is needed the first time to download build tools and dependencies.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\Build_Windows_EXE.ps1"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo BUILD FAILED. Check %%LOCALAPPDATA%%\GestureHMI\logs\build_last.log
  pause
  exit /b %RC%
)
echo.
echo Build finished. Windows EXE: dist\GestureHMI\GestureHMI.exe
echo Release package: GestureHMI_Windows_x64.zip
echo You may now double-click Install_Gesture_HMI.bat.
pause
exit /b 0
