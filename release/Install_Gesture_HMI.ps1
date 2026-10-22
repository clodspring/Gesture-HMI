# Install either a prebuilt release ZIP or a freshly built local Windows EXE.
# No administrator elevation or Python needed to RUN the installed application.
param([string]$SourceDirectory = '', [string]$LauncherFile = '')
$ErrorActionPreference='Stop'
if ($SourceDirectory) {
    $sourceDir = $SourceDirectory
} else {
    $sourceDir = Join-Path $PSScriptRoot 'GestureHMI'
}
$sourceExe=Join-Path $sourceDir 'GestureHMI.exe'
if (-not (Test-Path -LiteralPath $sourceExe -PathType Leaf)) {
    Write-Host 'ERROR: GestureHMI.exe not found beside the installer.' -ForegroundColor Red
    Write-Host 'For a source checkout, run the ROOT Install_Gesture_HMI.bat to build the EXE locally.'
    exit 2
}
if (-not $env:LOCALAPPDATA) { Write-Host 'ERROR: LOCALAPPDATA not set'; exit 2 }
$base=Join-Path $env:LOCALAPPDATA 'Programs'
$dest=Join-Path $base 'GestureHMI'
$backup=Join-Path $base 'GestureHMI.previous'
$settings=Join-Path $env:LOCALAPPDATA 'GestureHMI'
$process=Get-Process -Name 'GestureHMI' -ErrorAction SilentlyContinue
if ($process) {
    Write-Host 'Please close Gesture HMI before updating, then run the installer again.'
    exit 3
}
New-Item -ItemType Directory -Force -Path $base,$settings | Out-Null
$previousMoved=$false
try {
    if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Recurse -Force }
    if (Test-Path -LiteralPath $dest) {
        Move-Item -LiteralPath $dest -Destination $backup
        $previousMoved=$true
    }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Get-ChildItem -LiteralPath $sourceDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force -ErrorAction Stop
    }
    $startBat = if ($LauncherFile) { $LauncherFile } else { Join-Path $PSScriptRoot 'Start_Gesture_HMI.bat' }
    if (Test-Path -LiteralPath $startBat -PathType Leaf) {
        Copy-Item -LiteralPath $startBat -Destination (Join-Path $dest 'Start_Gesture_HMI.bat') -Force
    }
    if (-not (Test-Path -LiteralPath (Join-Path $dest 'Start_Gesture_HMI.bat') -PathType Leaf)) {
        throw 'Start_Gesture_HMI.bat not copied into install folder.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $dest 'GestureHMI.exe') -PathType Leaf)) {
        throw 'Failed to copy GestureHMI.exe.'
    }
    Write-Host 'Checking installed gesture and MediaPipe models (no webcam access)...'
    & (Join-Path $dest 'GestureHMI.exe') --doctor
    if ($LASTEXITCODE -ne 0) { throw 'Installed application did not pass the startup diagnostics.' }
    $ws=New-Object -ComObject WScript.Shell
    $desktop=[Environment]::GetFolderPath('DesktopDirectory')
    $programs=[Environment]::GetFolderPath('Programs')
    foreach ($folder in @($desktop,$programs)) {
        if (-not [string]::IsNullOrWhiteSpace($folder)) {
            $shortcut=$ws.CreateShortcut((Join-Path $folder 'Gesture HMI.lnk'))
            # Start through BAT so a startup error does not vanish instantly.
            $shortcut.TargetPath=Join-Path $dest 'Start_Gesture_HMI.bat'
            $shortcut.WorkingDirectory=$dest
            $shortcut.IconLocation=(Join-Path $dest 'GestureHMI.exe')
            $shortcut.Description='Gesture HMI - webcam gesture control'
            $shortcut.Save()
        }
    }
    if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Recurse -Force }
    Write-Host 'Gesture HMI installed successfully.' -ForegroundColor Green
    Write-Host ('Program: '+$dest)
    Write-Host ('Settings: '+$settings+' (preserved during updates)')
    Write-Host 'Double-click the desktop shortcut: Gesture HMI'
    exit 0
} catch {
    Write-Host ('INSTALL ERROR: '+$_.Exception.Message) -ForegroundColor Red
    try {
        if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
        if ($previousMoved -and (Test-Path -LiteralPath $backup)) {
            Move-Item -LiteralPath $backup -Destination $dest
            Write-Host 'Previous version restored.'
        }
    } catch { Write-Host 'WARNING: Could not restore previous installation.' }
    exit 1
}
