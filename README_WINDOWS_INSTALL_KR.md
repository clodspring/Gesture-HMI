# Gesture HMI v5: GitHub 없이 내 노트북에서 EXE를 만들고 설치하기

## 한 번만 할 것

Windows 10/11 x64 노트북에서 `Install_Gesture_HMI.bat`를 더블클릭하세요.

처음에는 자동으로 다음 과정을 진행합니다.

```
Windows ZIP 완전 압축 해제
   ↓
Install_Gesture_HMI.bat 더블클릭
   ↓
Python 3.12.10 빌드 도구 설치 (현재 사용자만, Conda 불필요)
   ↓
별도 Python 가상환경에 패키지 설치 (PyTorch 제외)
   ↓
MediaPipe hand_landmarker.task 다운로드
   ↓
NumPy MLP 및 제스처 기능 테스트
   ↓
PyInstaller로 GestureHMI.exe 생성
   ↓
EXE 자체의 모델·MediaPipe 로딩 진단
   ↓
프로그램 설치, 바탕화면 바로가기 생성
   ↓
바탕화면 Gesture HMI 더블클릭
```

빌드에 사용하는 Python은 `%LOCALAPPDATA%\GestureHMI\BuildTools`에 분리 보관합니다. 다른 사용자에게 배포되는 EXE에는 필요한 런타임이 자체 포함됩니다.

## 테스트: 빌드만 먼저 해보고 싶다면

`Build_Windows_EXE.bat` 더블클릭 → 빌드 성공 후 `dist\GestureHMI\GestureHMI.exe` 실행. 진단만 하려면 Windows CMD에서:

```bat
dist\GestureHMI\GestureHMI.exe --doctor
dist\GestureHMI\GestureHMI.exe --doctor --camera
```

첫 명령은 카메라를 열지 않고 모델과 라이브러리만 검사합니다. 두 번째 명령은 연결된 웹캠도 확인합니다.

## 다른 PC에 배포하기

빌드 성공 후 프로젝트 최상위에 `GestureHMI_Windows_x64.zip`이 만들어집니다. 이 파일만 전달하면 됩니다.

상대방은 ZIP을 풀고 `Install_Gesture_HMI.bat`를 더블클릭하면 됩니다. 이때 설치 BAT는 이미 생성된 EXE를 복사하므로 **다운로드, Conda/Python/pip 실행을 하지 않습니다.**

설치 프로그램은 `%LOCALAPPDATA%\Programs\GestureHMI`에 프로그램을 설치하고 바탕화면 및 시작 메뉴 바로가기를 생성합니다. 사용자 설정 `%LOCALAPPDATA%\GestureHMI`는 업데이트 시 유지합니다.

## 실패했을 때는 무엇을 확인하나요?

- `BUILD FAILED`: `%LOCALAPPDATA%\GestureHMI\logs\build_last.log` 확인
- 설치 실패: 설치 BAT 창에 표시되는 설치 오류 확인
- 실행/카메라/모델 문제: `%LOCALAPPDATA%\GestureHMI\logs\app_last.log` 확인

설치 PC의 정책이 외부 다운로드, Python 설치 또는 서명되지 않은 EXE 실행을 막으면 자동 빌드·실행에 실패할 수 있습니다. 이 경우 로그를 바탕으로 원인을 확인해야 하며 무조건 보안 제한을 우회하지 않습니다.

## 현재 검증 상태

이 ZIP은 Linux 환경에서 **NumPy 모델, 제스처 테스트, Python 정적 검사, 배포 구조 검사**까지 검증했습니다. Windows EXE 자체의 실제 빌드 및 Windows 설치까지 성공했다고 주장할 수는 없습니다. 최종 확인은 Windows 노트북에서 BAT를 실행했을 때 완료됩니다.
