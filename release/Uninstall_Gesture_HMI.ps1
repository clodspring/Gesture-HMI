$ErrorActionPreference='Stop'
if (-not $env:LOCALAPPDATA) { exit 2 }
$folder=Join-Path $env:LOCALAPPDATA 'Programs\GestureHMI'
$proc=Get-Process -Name GestureHMI -ErrorAction SilentlyContinue
if ($proc) { Write-Host 'Close Gesture HMI before uninstalling.'; exit 3 }
try {
    if (Test-Path -LiteralPath $folder) { Remove-Item -LiteralPath $folder -Force -Recurse }
    foreach ($targetFolder in @([Environment]::GetFolderPath('DesktopDirectory'),
                              [Environment]::GetFolderPath('Programs'))) {
        if ($targetFolder) {
            $shortcut=Join-Path $targetFolder 'Gesture HMI.lnk'
            if (Test-Path -LiteralPath $shortcut) { Remove-Item -LiteralPath $shortcut -Force }
        }
    }
    Write-Host 'Uninstalled. Settings remain in LOCALAPPDATA\GestureHMI.'
    exit 0
} catch { Write-Host ('Uninstall failed: '+$_.Exception.Message); exit 1 }
