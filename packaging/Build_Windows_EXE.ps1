# Gesture HMI Windows local build.
# Run through Build_Windows_EXE.bat / Install_Gesture_HMI.bat on Windows 10/11 x64.
# Build tools live outside the project and never enter the final EXE ZIP.
param([switch]$SkipZip)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is not available.' }
$toolRoot = Join-Path $env:LOCALAPPDATA 'GestureHMI\BuildTools'
$logDir = Join-Path $env:LOCALAPPDATA 'GestureHMI\logs'
New-Item -ItemType Directory -Path $toolRoot,$logDir -Force | Out-Null
$logPath = Join-Path $logDir 'build_last.log'
try { Start-Transcript -Path $logPath -Force -ErrorAction Stop | Out-Null } catch { Write-Warning "Could not start transcript: $_" }

function Step([string]$message) { Write-Host "`n=== $message ===" -ForegroundColor Cyan }
function Require-Ok([string]$task) {
    if ($LASTEXITCODE -ne 0) { throw "$task failed with exit code $LASTEXITCODE. Build log: $logPath" }
}
function Download-Verified([string]$url,[string]$out,[string]$sha256) {
    if (Test-Path -LiteralPath $out -PathType Leaf) {
        $got=(Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash
        if ($got -ieq $sha256) { return }
        Remove-Item -LiteralPath $out -Force
    }
    Write-Host "Downloading official file: $url"
    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -TimeoutSec 240
    $got=(Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash
    if ($got -ine $sha256) {
        Remove-Item -LiteralPath $out -Force
        throw "Downloaded file SHA256 mismatch. Expected $sha256, got $got"
    }
}

try {
    if ([Environment]::Is64BitOperatingSystem -ne $true) { throw 'This builder needs Windows x64.' }
    if ($env:OS -ne 'Windows_NT') { throw 'Run on Windows 10/11 x64; PyInstaller cannot cross-build Windows EXE on Linux.' }
    if (-not (Test-Path (Join-Path $root 'gesture_pointer_step19_v3b_portable.py'))) { throw 'Gesture HMI source file not found.' }
    if (-not (Test-Path (Join-Path $root 'models_v2\gesture_mlp_v2.npz'))) { throw 'NumPy model is missing.' }

    # An isolated known-good build Python avoids the user's broken Conda interpreter.
    $pythonHome = Join-Path $toolRoot 'Python312'
    $basePython = Join-Path $pythonHome 'python.exe'
    if (-not (Test-Path -LiteralPath $basePython)) {
        Step '1/7 - Download verified Python 3.12.10 (build-only; no Conda)'
        $installer = Join-Path $toolRoot 'python-3.12.10-amd64.exe'
        # SHA-256 of official python.org Python 3.12.10 amd64 installer.
        Download-Verified 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' `
            $installer '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'
        $arguments = '/quiet InstallAllUsers=0 Include_launcher=0 PrependPath=0 Include_pip=1 Include_test=0 Include_doc=0 TargetDir="' + $pythonHome + '"'
        Write-Host ('Installing build-only Python at: '+$pythonHome)
        $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -notin @(0,3010)) { throw "Python installer failed (exit code $($process.ExitCode))." }
        if (-not (Test-Path -LiteralPath $basePython)) { throw 'Python installation finished without python.exe. Check Windows installer policy.' }
    } else { Step '1/7 - Reuse isolated Python 3.12' }
    & $basePython -c 'import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32; print(sys.version)'
    Require-Ok 'Python runtime validation'

    Step '2/7 - Prepare isolated package environment'
    $venv = Join-Path $toolRoot 'venv312'
    $python = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & $basePython -m venv $venv
        Require-Ok 'venv creation'
    }
    & $python -c 'import sys; print("Build interpreter:", sys.executable)'
    Require-Ok 'venv validation'
    & $python -m pip install --disable-pip-version-check -r (Join-Path $root 'requirements-windows.txt')
    Require-Ok 'Dependency installation'
    & $python -m pip check
    Require-Ok 'Dependency compatibility check'
    & $python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None; print("No PyTorch in build environment")'
    Require-Ok 'Torch exclusion check'

    Step '3/7 - Download official hand model once'
    $assets = Join-Path $root 'assets'
    New-Item -ItemType Directory -Path $assets -Force | Out-Null
    $task = Join-Path $assets 'hand_landmarker.task'
    if (-not (Test-Path $task) -or (Get-Item $task).Length -lt 100000) {
        $temp = "$task.download"
        try {
            Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' `
                -OutFile $temp -UseBasicParsing -TimeoutSec 240
            if ((Get-Item $temp).Length -lt 100000) { throw 'MediaPipe model was empty or too small.' }
            Move-Item -LiteralPath $temp -Destination $task -Force
        } finally { if (Test-Path $temp) { Remove-Item $temp -Force } }
    }
    Write-Host ('Hand model bytes: ' + (Get-Item $task).Length)

    Step '4/7 - Run source model and application checks'
    Push-Location $root
    try {
        & $python test_portability.py
        Require-Ok 'Portability tests'
        & $python test_numpy_mlp.py
        Require-Ok 'NumPy MLP tests'
        & $python test_distribution.py
        Require-Ok 'Distribution tests'
        & $python gesture_pointer_step19_v3b_portable.py --self-test
        Require-Ok 'Gesture regression tests'
        & $python gesture_pointer_step19_v3b_portable.py --doctor
        Require-Ok 'Source model/runtime doctor'

        Step '5/7 - Build PyInstaller standalone onedir EXE'
        & $python -m PyInstaller --noconfirm --clean --onedir --console --name GestureHMI `
          --collect-all mediapipe --hidden-import mediapipe.tasks.c `
          --collect-submodules pycaw --collect-submodules uiautomation `
          --collect-submodules comtypes --hidden-import win32api `
          --exclude-module torch --exclude-module torchvision --exclude-module torchaudio `
          --add-data 'models_v2\gesture_mlp_v2.npz;models_v2' `
          --add-data 'assets\hand_landmarker.task;assets' `
          gesture_pointer_step19_v3b_portable.py
        Require-Ok 'PyInstaller build'
        $exe = Join-Path $root 'dist\GestureHMI\GestureHMI.exe'
        if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'PyInstaller reported success but EXE was not produced.' }

        Step '6/7 - Validate Windows EXE and bundled models'
        & $exe --resource-check
        Require-Ok 'EXE resource check'
        & $exe --self-test
        Require-Ok 'EXE regression tests'
        & $exe --doctor
        Require-Ok 'EXE MediaPipe/Windows doctor'
        if (Test-Path 'dist\GestureHMI\_internal\torch') { throw 'Torch unexpectedly present in bundle.' }

        Step '7/7 - Create install-ready Windows release ZIP'
        $releaseOut = Join-Path $root 'release_out'
        $bundleOut = Join-Path $releaseOut 'GestureHMI'
        if (Test-Path -LiteralPath $releaseOut) { Remove-Item -LiteralPath $releaseOut -Recurse -Force }
        New-Item -ItemType Directory -Path $bundleOut -Force | Out-Null
        Copy-Item -Path (Join-Path $root 'dist\GestureHMI\*') -Destination $bundleOut -Recurse -Force
        foreach ($filename in @('Install_Gesture_HMI.bat','Start_Gesture_HMI.bat','README_WINDOWS_INSTALL_KR.md')) {
            Copy-Item -LiteralPath (Join-Path $root $filename) -Destination (Join-Path $releaseOut $filename) -Force
        }
        foreach ($filename in @('Install_Gesture_HMI.ps1','Uninstall_Gesture_HMI.bat','Uninstall_Gesture_HMI.ps1')) {
            Copy-Item -LiteralPath (Join-Path (Join-Path $root 'release') $filename) -Destination (Join-Path $releaseOut $filename) -Force
        }
        if (-not $SkipZip) {
            $archive = Join-Path $root 'GestureHMI_Windows_x64.zip'
            if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
            Compress-Archive -Path (Join-Path $releaseOut '*') -DestinationPath $archive -CompressionLevel Optimal -Force
            $hash=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
            Set-Content -LiteralPath (Join-Path $root 'SHA256SUMS_release.txt') -Encoding ascii -Value "$hash  GestureHMI_Windows_x64.zip"
            Write-Host "READY ZIP: $archive" -ForegroundColor Green
        }
        Write-Host "READY EXE: $exe" -ForegroundColor Green
    } finally { Pop-Location }
    exit 0
} catch {
    Write-Host "`nBUILD FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Build log: $logPath" -ForegroundColor Yellow
    exit 1
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}
