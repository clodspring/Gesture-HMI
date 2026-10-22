# STEP 19 최종 코드 및 Windows 배포 구조

> **기준일:** 2026-10-22  
> **기능 버전:** STEP 19 v3b 계열 (v3a 음성 안정화 포함)  
> **패키징 버전:** Gesture HMI v5 Windows release builder (소스)  
> **중요:** 현재 GitHub에 있는 것은 **최종 기능 소스 + EXE 빌드 소스**다. Windows용 `GestureHMI.exe`가 정식 배포되었다고 의미하지 않는다.

## 1. 소스코드 바로 보기

**[STEP 19 최종 Python 코드 전체 보기](../gesture_pointer_step19_v3b_portable.py)**

GitHub는 `.py` 파일을 코드 뷰어로 보여주므로 Markdown 문서 안에 5천 줄 넘는 코드를 복사하지 않아도 브라우저에서 바로 열람할 수 있다.

| 파일 | 용도 |
|---|---|
| [`gesture_pointer_step19_v3b_portable.py`](../gesture_pointer_step19_v3b_portable.py) | **최종 HMI 기능 소스**: 웹캠, 인식, 상태 제어, Windows 제어, Handy 연동, 진단 |
| [`models_v2/gesture_mlp_v2.npz`](../models_v2/gesture_mlp_v2.npz) | 배포 시 사용할 학습된 NumPy MLP 가중치 |
| [`requirements-windows.txt`](../requirements-windows.txt) | Windows 빌드 환경에 설치할 Python 패키지 목록 (PyTorch 제외) |
| [`packaging/Build_Windows_EXE.ps1`](../packaging/Build_Windows_EXE.ps1) | PyInstaller 기반 EXE 및 배포 ZIP 빌드 |
| [`release/Install_Gesture_HMI.ps1`](../release/Install_Gesture_HMI.ps1) | 빌드된 EXE를 사용자 폴더에 설치 |
| [`release/Uninstall_Gesture_HMI.ps1`](../release/Uninstall_Gesture_HMI.ps1) | 설치 제거 |
| [`.github/workflows/windows-release.yml`](../.github/workflows/windows-release.yml) | GitHub Windows 빌드·설치/제거 테스트 및 아티팩트 생성 |
| [`test_numpy_mlp.py`](../test_numpy_mlp.py), [`test_portability.py`](../test_portability.py), [`test_distribution.py`](../test_distribution.py) | 소스 및 배포 구조 검사 |

### 최종 기능 소스 무결성 식별값

- 파일명: `gesture_pointer_step19_v3b_portable.py`
- 줄 수: **5,194줄** (현재 업로드된 v5 원본 기준)
- SHA-256: `79cc295c82bc9cdb1bce5ff3718b9cc133eb1d1d204a0121272415dbd0df208f`
- NumPy 모델 SHA-256: `9140aff98cac7731d29b03c7856f66b0fc8ec89eff984485cc313622c6d7b3f5`

SHA-256은 현재 확인된 파일을 식별하기 위한 값이다. 코드를 수정하면 값이 바뀐다.

## 2. 최종 소프트웨어 구조

```text
[Webcam]
   │ OpenCV, camera auto scan (index 0..4)
   ▼
[MediaPipe Hand Landmarker]
   │ 21개 hand landmarks
   ▼
[Feature extraction & normalization]
   ├────► [NumPy MLP] ──► peace / one / stop / no_gesture
   │
   └────► [Geometry / trajectory / duration gates]
                     ├► Pinch / click / swipe / circle
                     ├► Thumbs-up Hold-to-Talk
                     ├► Fist -> OK clear-text
                     └► Gun volume and media seek
   ▼
[HandsFreeController]
  LOCKED / ACTIVE / PAUSED / STANDBY
   ▼
[Windows control / UIAutomation / audio / media]
   └────► [Handy Ctrl+Space] (선택적 외부 설치)
```

- **손 추적:** `MediaPipe` 21개 손 랜드마크 추정
- **정적 자세:** 학습된 MLP `42 → 64 → 32 → 4` (NumPy 추론)
- **복합 제스처:** 기하·시간·이동 궤적을 이용해 판정하는 하이브리드 구조
- **안전 장치:** 활성 상태, 최근 클릭·입력창 포커스 검증, 재작동 시간 제한, 카메라 신호 소실 처리

## 3. 주요 STEP 19 기능

| 기능 | 입력·동작 | 주요 구현 위치 |
|---|---|---|
| HMI 활성화 / 잠금 | Peace 유지로 ACTIVE, Stop으로 잠금 | `HandsFreeController` |
| 포인터 | One과 손가락 랜드마크를 화면 좌표로 변환 | `to_features`, `LatestMouseWorker` 등 |
| 클릭 / 스크롤 | 손가락 접촉·Pinch 기하 및 제스처 궤적 | `click_metrics`, `PinchEntryGate`, 스크롤 함수 |
| 뒤로 가기 / Enter | Pinch 이동 방향과 이력 기반 명령 | `back_metrics`, `enter_metrics` |
| 한국어 받아쓰기 | Thumbs-up 유지와 Handy 단축키 연동 | `ThumbPressGate`, `HandyHoldKeys` |
| 텍스트 삭제 | Fist → OK 순서와 포커스 안전 조건 | `ClearTextGate`, `FocusedEditGuard` |
| 미디어 재생/일시정지 | 원형 손동작 및 미디어 명령 | `circle_metrics`, `MediaKeyWorker` |
| 볼륨 | Gun 손 모양의 세로 이동 | `gun_volume_geometry`, `VolumeMotion` |
| 영상 ±10초 | Gun 손 모양의 좌우 이동 | `GunSeekMotion`, `WindowsMediaSeek` |
| 카메라 자동 선택 | 카메라 index 0~4와 백엔드 순차 검사 | `open_compatible_camera` |

**제한:** Windows 미디어 세션 제어는 선택적 `winrt` 패키지의 영향을 받을 수 있다. 특정 YouTube 화면에서 J/L 단축키 대체 경로가 구현되어 있지만, 모든 플레이어나 입력 상태에서 성공한다고 단정할 수 없다.

## 4. 추론 경량화: PyTorch 제거

최종 사용자 실행 경로는 `.pth` 체크포인트가 아니라 [`gesture_mlp_v2.npz`](../models_v2/gesture_mlp_v2.npz) 파일을 불러온다.

```python
# 실제 핵심 함수명 (최종 소스에서 확인됨)
weights, mean, std, clip, classes = load_numpy_mlp()
probs = numpy_mlp_probs(features, weights)
```

`load_numpy_mlp()`는 `W1, b1, W2, b2, W3, b3` 배열의 형상과 클래스명을 검증하고, `numpy_mlp_probs()`는 두 ReLU 은닉층과 Softmax 연산을 수행한다. 배포 Python 런타임에는 `import torch`가 없다.

## 5. 배포 준비와 실제 배포는 다름

### A. 개발·빌드용 저장소 (이 GitHub 저장소)

- Python 최종 코드와 NumPy 모델 제공
- GitHub Actions에서 Windows EXE를 생성하기 위한 스크립트 제공
- `Code → Download ZIP`으로 받는 것은 **소스 ZIP**이고, 이것을 다운로드했다고 곧바로 `GestureHMI.exe`가 생기는 것은 아님
- 개발자용 로컬 빌드에는 독립된 Python 빌드 도구 설치가 필요할 수 있음

### B. 일반 사용자용 Windows 실행파일 (빌드 성공 후)

GitHub Actions에서 빌드가 성공하면 작업 실행 페이지의 **Artifacts**에서 Windows 아티팩트를 받을 수 있다. 최종 배포 예정 파일은 `GestureHMI_Windows_x64.zip`이다.

이 ZIP 내부의 설치 스크립트는 이미 묶인 EXE를 설치하는 것이며, 일반 사용자 PC에 Python·Conda·PyTorch 설치를 요구하지 않도록 설계됐다.

**현재 확인 상태:** 최종 Windows EXE가 실제로 빌드·설치·실행되었는지에 대한 검증 증거는 아직 이 문서에 반영되지 않았다. 빌드 성공 후에만 사용자용 설치방법과 다운로드 주소를 공개 문서에 확정하는 것을 권장한다.

## 6. 진단 명령 (EXE 빌드 성공 후 Windows에서)

```bat
GestureHMI.exe --doctor
GestureHMI.exe --doctor --camera
GestureHMI.exe --camera-test
GestureHMI.exe --self-test
```

- `--doctor`: NumPy 모델, MediaPipe 모델, 주요 Python 모듈 검증 (카메라 테스트는 기본 생략)
- `--doctor --camera`: 카메라 열기까지 검사
- `--camera-test`: 카메라 프레임 확인/미리보기
- `--self-test`: 코드에 포함된 주요 로직 회귀 테스트

`--doctor` 통과는 모델·의존성 점검에 해당하며, 실제로 포인터·미디어·Handy가 사용자 의도대로 움직이는지까지 검증하는 결과는 아니다.

## 7. 배포 승인 전에 확인할 점

- Windows 10/11 x64 환경에서 PyInstaller 빌드 성공
- Python/Conda가 없는 다른 PC에서 설치 및 재시작 성공
- 카메라 권한 및 USB/내장 카메라 번호 차이 확인
- 주요 제스처 기능(클릭, 스크롤, 텍스트 편집, 음성 등) 실기 테스트
- 별도 Handy 설치와 Ctrl+Space **길게 누르기** 설정
- Windows 보안정책/SmartScreen 대응
- 선택적 WinRT 미디어 기능 확인
- HaGRIDv2, MediaPipe, Handy, 학습 가중치 등의 라이선스·재배포 조건 확인

## 8. 함께 보면 좋은 문서

- [Gesture HMI 개발 기록](DEVELOPMENT_HISTORY.md)
- [프로젝트 README](../README.md)
- [Windows 사용자/로컬 빌드 설명](../README_WINDOWS_INSTALL_KR.md)
