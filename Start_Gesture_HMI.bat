@echo off
setlocal
set "APP="
if exist "%~dp0GestureHMI.exe" set "APP=%~dp0GestureHMI.exe"
if not defined APP if exist "%~dp0GestureHMI\GestureHMI.exe" set "APP=%~dp0GestureHMI\GestureHMI.exe"
if not defined APP if exist "%~dp0dist\GestureHMI\GestureHMI.exe" set "APP=%~dp0dist\GestureHMI\GestureHMI.exe"
if not defined APP if exist "%LOCALAPPDATA%\Programs\GestureHMI\GestureHMI.exe" set "APP=%LOCALAPPDATA%\Programs\GestureHMI\GestureHMI.exe"
if not defined APP if exist "%~dp0gesture_pointer_step19_v3b_portable.py" (
  echo EXE not built yet - launching one-click local build and installation.
  call "%~dp0Install_Gesture_HMI.bat"
  if errorlevel 1 exit /b 1
  if exist "%LOCALAPPDATA%\Programs\GestureHMI\GestureHMI.exe" set "APP=%LOCALAPPDATA%\Programs\GestureHMI\GestureHMI.exe"
)
if not defined APP (
  echo ERROR: GestureHMI.exe is missing. Run Install_Gesture_HMI.bat in the Windows release ZIP.
  pause
  exit /b 2
)
echo === Gesture HMI Windows ===
echo Running: "%APP%"
"%APP%" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
 echo Gesture HMI ended with error code %RC%.
 echo Log: %%LOCALAPPDATA%%\GestureHMI\logs\app_last.log
 pause
)
exit /b %RC%
