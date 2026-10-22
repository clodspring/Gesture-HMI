@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install_Gesture_HMI.ps1"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo Gesture HMI is ready. Use the new desktop shortcut.
) else (
  echo Installation failed. See the message above.
)
pause
exit /b %RC%
