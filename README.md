# Gesture HMI — Hand Gesture Control for Windows

**MediaPipe + HaGRIDv2-landmark MLP + NumPy + Windows control + optional Handy dictation**

웹캠 기반 비접촉 인터페이스로 Windows 포인터·클릭·스크롤·텍스트 편집·미디어 조작을 제어하는 팀 개발 프로젝트입니다. STEP 1부터 STEP 19까지 인식과 상호작용을 개선하고, 현재는 다른 PC에서도 설치할 수 있도록 Windows EXE 배포를 준비하고 있습니다.

> **현재 배포 상태 (2026-10-22):** STEP 19 최종 Python **소스코드**와 v5 Windows **EXE 빌드 소스**가 공개되어 있습니다. GitHub Actions에서 빌드된 Windows EXE의 실제 설치·실기 검증이 완료됐다는 의미는 아닙니다. `Code → Download ZIP`은 **소스 ZIP**입니다.

## Demo Video

Watch Gesture HMI in action.

Real-time hand gesture control for cursor movement,
volume adjustment, voice-assisted search, and browser navigation.


https://github.com/user-attachments/assets/ba776116-d439-4863-888b-a1ad2dfd836e


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

## Team & Contributions

**팀명: 핑거 프린세스 (Finger Princess)**

Gesture HMI는 비접촉 Windows 제어라는 공동 목표 아래 **인터랙션 기획 → AI 기반 제스처 인식 및 PoC 구현 → 실사용 검증 → 발표·시연**으로 역할을 나누어 진행한 팀 프로젝트입니다.

| 팀원 | 주요 담당 | 세부 역할 및 기여 |
|---|---|---|
| **이찬샘** | **Technical Implementation · PoC Development** | MediaPipe 손 랜드마크와 MLP 기반 제스처 인식을 Windows 조작 기능에 연결하는 **PoC 최종 구현** 담당. 포인터·클릭·스크롤·음성 입력·미디어 제어의 통합, 제스처 상태 전이 및 오작동 방지 로직 구현, 최종 소스 정리와 Windows 배포 준비. |
| **이승준** | **Presentation · STT Validation** | **Handy(Whisper Small) 음성 입력 테스트** 담당. 제스처와 음성 입력을 결합한 사용 시나리오 점검에 참여하고, 기술 구성·모델 성능·실사용 검증 결과를 이해하기 쉽게 전달할 **발표자료 제작 및 시각화** 담당. |
| **장기준** | **Project Planning · Interaction Design · Presentation Flow** | **제스처별 액션 선정과 사용자 조작 흐름 설계** 담당. 조리 중 비접촉 제어 시나리오를 중심으로 기능 사용 순서를 구체화하고, 문제 정의 → 구현 방식 → 검증 결과 → 한계·개선 방향으로 이어지는 **발표 스토리 구성 및 발표 준비** 담당. |

### 공동 수행 및 의사결정

- **기술·데이터셋 탐색 및 선정:** MediaPipe 기반 손 인식, HaGRIDv2 랜드마크 데이터, MLP 제스처 분류, Handy/Whisper Small 음성 입력의 적용 방향을 함께 논의했습니다. 당시 기록에 개인별 모델 비교 책임자는 명시되어 있지 않으므로 팀 공동 기여로 표기합니다.
- **PoC 기능 협의:** 포인터만으로 검색할 때의 불편함을 확인하고, 음성 검색을 결합하는 방향을 논의했습니다.
- **사용성 평가 및 개선 논의:** 맨손·비닐장갑·검정 라텍스 장갑, 조명 ON/OFF 등 조건에서 발생한 제스처 인식 한계를 검토하고, 오작동 및 향후 개선 방향을 함께 정리했습니다.
- **최종 시연 준비:** 손 제스처와 음성 검색을 결합한 실제 Windows 사용 흐름을 점검하고 발표 시연을 준비했습니다.

> 역할 표기는 2026-10-16 및 2026-10-20 팀활동 기록에 명시된 담당 업무를 중심으로 세분화한 것입니다. 모델 탐색·실험 등 개인별 기여가 명확히 나뉘지 않은 항목은 공동 작업으로 표시했습니다.

