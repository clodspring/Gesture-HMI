# Gesture HMI v5 — Windows 로컬 원클릭 빌드·설치

**일반 사용자용 Windows EXE는 Python, Conda, PyTorch 없이 실행됩니다.**

이 프로젝트는 기존 STEP 19 제스처·음성 로직과 NumPy MLP 모델을 보존하면서, **GitHub에 업로드하지 않고 Windows 노트북에서 배포용 EXE를 만드는 소스 패키지**입니다.

## 내 노트북에서 처음 설치하기

1. 이 ZIP을 `압축 풀기`로 완전히 해제합니다. ZIP 미리보기 내부에서 BAT를 실행하지 마세요.
2. 폴더 맨 위에 있는 **`Install_Gesture_HMI.bat`**를 더블클릭합니다.
3. 첫 실행에만 인터넷으로 **빌드 전용 Python 3.12.10**, 라이브러리, MediaPipe 손 모델을 다운로드합니다. 기존 Conda 환경을 사용하거나 변경하지 않습니다.
4. 소스 검증 → PyInstaller EXE 생성 → EXE 모델 검증 → 설치까지 자동 진행합니다.
5. 성공하면 바탕화면 **Gesture HMI** 바로가기로 실행합니다.

별도로 Build, Test Camera, Link Conda를 실행할 필요가 없습니다. 처음부터 `Start_Gesture_HMI.bat`를 누르더라도 EXE가 없다면 빌드와 설치로 연결되도록 했습니다.

## 파일 역할

| 파일 | 동작 |
|---|---|
| `Install_Gesture_HMI.bat` | **권장**: EXE가 없으면 로컬 빌드부터 자동 실행한 뒤 설치 |
| `Start_Gesture_HMI.bat` | EXE 실행. EXE가 없고 소스만 있을 때 설치기로 안내·연결 |
| `Build_Windows_EXE.bat` | 개발자용: 설치 없이 Windows EXE 및 배포 ZIP만 생성 |
| `packaging/Build_Windows_EXE.bat` | 위 빌드 BAT와 동일하게 작동하는 호환 진입점 |
| `release/Install_Gesture_HMI.ps1` | 생성된 EXE를 사용자별 설치 위치로 복사·진단·바로가기 생성 |

## 최종 생성되는 파일

- `dist\GestureHMI\GestureHMI.exe` — 로컬 테스트용 실행 파일
- `GestureHMI_Windows_x64.zip` — 다른 사용자에게 **이 파일만** 전달
- `%LOCALAPPDATA%\Programs\GestureHMI\GestureHMI.exe` — 사용자 PC에 설치된 프로그램
- `%LOCALAPPDATA%\GestureHMI\logs\build_last.log` — 빌드 문제 진단 로그
- `%LOCALAPPDATA%\GestureHMI\logs\app_last.log` — 실행 중 오류 진단 로그

최종 ZIP에는 설치 스크립트와 EXE 및 모델·DLL이 포함되며 **빌드 전용 Python 환경은 포함되지 않습니다.**

## 중요 조건

- **빌드 PC**: Windows 10/11 **x64**, 인터넷, 다운로드·실행이 허용되는 보안 정책. 첫 설치 시 빌드 전용 Python과 pip 패키지 설치를 진행합니다.
- **최종 사용자 PC**: Windows 10/11 x64, 웹캠, 해당 OS가 필요로 하는 시스템 런타임. Python/Conda/PyTorch는 필요 없습니다.
- **Handy**: 음성입력만을 위한 선택적 외부 프로그램입니다. Handy가 없어도 나머지 손 제스처 기능은 실행되도록 설계되었습니다.
- **개인정보**: 카메라 영상 전송 기능을 별도로 추가하지 않았습니다. 다만 설치·진단 로그에는 로컬 Windows 사용자 경로가 들어갈 수 있습니다.
- **처음 시도에는 디스크 여유 공간이 필요합니다.** 빌드 라이브러리, MediaPipe 및 EXE 번들이 저장됩니다. 충분한 공간을 확보해 주세요.
- PyInstaller가 실제 Windows에서 성공하는지는 이 환경에서 실행 검증을 할 수 없었습니다. 오류가 있다면 `build_last.log` 끝부분으로 원인을 특정할 수 있습니다.

자세한 절차는 [`README_WINDOWS_INSTALL_KR.md`](README_WINDOWS_INSTALL_KR.md) 참고.
