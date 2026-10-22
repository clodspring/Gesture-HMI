# Gesture HMI — Hand Gesture Control for Windows

**MediaPipe + HaGRIDv2-landmark MLP + NumPy + Windows control + optional Handy dictation**

웹캠 기반 비접촉 인터페이스로 Windows 포인터·클릭·스크롤·텍스트 편집·미디어 조작을 제어하는 개인 개발 프로젝트입니다. STEP 1부터 STEP 19까지 인식과 상호작용을 개선하고, 현재는 다른 PC에서도 설치할 수 있도록 Windows EXE 배포를 준비하고 있습니다.

> **현재 배포 상태 (2026-10-22):** STEP 19 최종 Python **소스코드**와 v5 Windows **EXE 빌드 소스**가 공개되어 있습니다. GitHub Actions에서 빌드된 Windows EXE의 실제 설치·실기 검증이 완료됐다는 의미는 아닙니다. `Code → Download ZIP`은 **소스 ZIP**입니다.

## 바로가기

| 문서 / 코드 | 내용 |
|---|---|
| **[개발 과정: STEP 1~19](docs/DEVELOPMENT_HISTORY.md)** | 단계별 개발 이력, 문제와 개선, 모델 성능, Windows 배포까지의 변천 |
| **[STEP 19 최종 코드 설명](docs/STEP19_FINAL_CODE.md)** | 기능별 구현 방식, 파일 구조, NumPy 추론, 실행 및 진단, 배포 상태 |
| **[STEP 19 전체 Python 소스](gesture_pointer_step19_v3b_portable.py)** | 실제 실행 로직을 포함한 5,194줄 코드; GitHub에서 직접 열람 가능 |
| [MLP v2 가중치](models_v2/gesture_mlp_v2.npz) | 학습된 NumPy 모델 |
| [Windows 빌드·설치 안내](README_WINDOWS_INSTALL_KR.md) | 소스에서 EXE를 만드는 법과 사용자 설치 개요 |
| [GitHub Actions Windows 빌드](.github/workflows/windows-release.yml) | Windows 빌드/진단/아티팩트 자동화 코드 |

## 주요 기능

- **Hand tracking:** MediaPipe 21개 랜드마크
- **AI 분류:** `42 → 64 → 32 → 4` MLP (`peace`, `one`, `stop`, `no_gesture`) 및 손동작 기하·시간 규칙
- **Windows 제어:** 포인터, 클릭, 스크롤, 뒤로 가기, Enter, 텍스트 편집
- **미디어 제어:** 원형 동작 재생/일시정지, Gun 동작 볼륨 및 좌우 ±10초 탐색
- **한국어 받아쓰기:** Thumbs-up과 별도 Handy 프로그램 연동
- **배포 개선:** 카메라 자동 탐색(0~4), NumPy 런타임으로 PyTorch 추론 의존성 제거

손동작이 사라질 때의 입력 차단, 텍스트 편집 대상의 포커스 확인, 잘못된 반복 입력 방지 등 사용 안전성을 위한 상태 관리 로직도 포함되어 있습니다.

## 파일 구조

```text
Gesture-HMI/
├── .github/workflows/windows-release.yml
├── docs/
│   ├── DEVELOPMENT_HISTORY.md
│   └── STEP19_FINAL_CODE.md
├── models_v2/gesture_mlp_v2.npz
├── packaging/Build_Windows_EXE.ps1
├── release/Install_Gesture_HMI.ps1
├── gesture_pointer_step19_v3b_portable.py
├── requirements-windows.txt
├── test_numpy_mlp.py
├── test_portability.py
├── test_distribution.py
├── README_WINDOWS_INSTALL_KR.md
└── README.md
```

## Windows 사용자에게 안내할 내용

**소스 ZIP과 실행파일 배포 ZIP은 다릅니다.** 실제 빌드가 성공한 Windows 패키지는 GitHub의 Actions 아티팩트 또는 검증 완료 후 Releases에 별도로 게시할 예정입니다. 일반 사용자용 배포 파일에는 `GestureHMI.exe`와 설치용 BAT가 있어야 합니다.

Hands-free 제스처는 Windows PC의 웹캠 권한·보안정책에 영향을 받을 수 있으며, 한국어 음성 받아쓰기는 **Handy 외부 설치 및 Ctrl+Space 길게 누르기 설정**이 필요합니다.

## 테스트 및 제한

- 초기 랜드마크 MLP v1 정적 테스트 정확도는 **99.58%**로 기록되어 있으나, 실제 웹캠 전체 동작 성공률을 의미하지 않습니다.
- 추론 경량화 시 별도 샘플 **1,500 / 1,500**에서 PyTorch와 NumPy의 예측 클래스가 일치했습니다.
- 최종 STEP 19 소스에는 진단 및 회귀 테스트가 포함되어 있지만, 실제 배포 EXE·실물 카메라·다른 사용자 PC에서의 기능 검증은 별도입니다.
- Windows 미디어 세션의 일부 조작은 선택적 WinRT 패키지에 영향을 받으며, 앱별 지원 차이가 있습니다.

## 라이선스 및 외부 구성요소

프로젝트 소스와 학습 가중치에 적용할 공개 라이선스, HaGRIDv2·MediaPipe 모델 및 Handy 관련 재배포 조건은 정식 배포 전에 확인해야 합니다. 이 저장소에 LICENSE 파일이 없다면 별도의 오픈소스 사용 권한이 자동으로 부여된 것은 아닙니다.
