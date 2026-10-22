"""Static tests for Windows local build and release packaging (Linux-compatible)."""
from pathlib import Path
import ast
import zipfile

root=Path(__file__).resolve().parent
app=(root/'gesture_pointer_step19_v3b_portable.py').read_text(encoding='utf-8')
requirements=(root/'requirements-windows.txt').read_text(encoding='utf-8')
builder=(root/'packaging/Build_Windows_EXE.ps1').read_text(encoding='utf-8')
installer_bat=(root/'Install_Gesture_HMI.bat').read_text(encoding='utf-8')
installer_ps=(root/'release/Install_Gesture_HMI.ps1').read_text(encoding='utf-8')
startup=(root/'Start_Gesture_HMI.bat').read_text(encoding='utf-8')
old_builder=(root/'packaging/Build_Windows_EXE.bat').read_text(encoding='utf-8')
assert 'import torch' not in app and 'from torch' not in app
assert all(not line.strip().lower().startswith(('torch','pytorch')) for line in requirements.splitlines())
assert (root/'models_v2/gesture_mlp_v2.npz').exists()
assert not (root/'models_v2/gesture_mlp.pth').exists()
assert '--collect-all mediapipe' in builder
assert '--exclude-module torch' in builder
assert 'models_v2\\gesture_mlp_v2.npz;models_v2' in builder
assert 'assets\\hand_landmarker.task;assets' in builder
assert 'python-3.12.10-amd64.exe' in builder
assert 'Get-FileHash' in builder
assert 'TargetDir=' in builder
assert '-m venv' in builder
assert 'GestureHMI_Windows_x64.zip' in builder
assert 'Install_Gesture_HMI.ps1' in builder
assert 'call "%~dp0..\\Build_Windows_EXE.bat"' in old_builder
assert 'Build_Windows_EXE.ps1' in installer_bat
assert 'SourceDirectory' in installer_ps
assert 'GestureHMI\\GestureHMI.exe' in installer_bat
assert 'Install_Gesture_HMI.bat' in startup
assert 'GestureHMI.exe' in startup
assert 'conda activate' not in startup and 'conda activate' not in builder.lower()
assert '--doctor' in app and 'app_last.log' in app
ast.parse(app)
print('DISTRIBUTION AND LOCAL BUILD STRUCTURE: PASS')
