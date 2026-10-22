@echo off
setlocal
cd /d "%~dp0"
echo === Gesture HMI - One-Click Windows Installer ===
echo.
if exist "%~dp0GestureHMI\GestureHMI.exe" goto ReleaseInstall
if exist "%~dp0gesture_pointer_step19_v3b_portable.py" goto LocalInstall
echo ERROR: Neither packaged EXE nor Gesture HMI source code is present.
pause
exit /b 2

:ReleaseInstall
if not exist "%~dp0Install_Gesture_HMI.ps1" (
  echo ERROR: Release installation script missing.
  pause
  exit /b 2
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install_Gesture_HMI.ps1"
set "RC=%ERRORLEVEL%"
goto Result

:LocalInstall
if not exist "%~dp0dist\GestureHMI\GestureHMI.exe" (
    echo No EXE yet. Building locally, then installing...
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\Build_Windows_EXE.ps1"
    if errorlevel 1 goto BuildFailed
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0release\Install_Gesture_HMI.ps1" -SourceDirectory "%~dp0dist\GestureHMI" -LauncherFile "%~dp0Start_Gesture_HMI.bat"
set "RC=%ERRORLEVEL%"
goto Result

:BuildFailed
echo ERROR: Local build failed. Log: %LOCALAPPDATA%\GestureHMI\logs\build_last.log
pause
exit /b 1

:Result
if "%RC%"=="0" (
  echo SUCCESS - The Gesture HMI desktop shortcut is ready.
) else (
  echo INSTALL FAILED with error code %RC%.
)
pause
exit /b %RC%
