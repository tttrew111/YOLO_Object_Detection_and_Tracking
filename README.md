# YOLO 객체 탐지 및 ID 트래킹

드론에 장착된 카메라가 약 45도 각도로 내려다보는 영상에서 사람을 탐지하고, 사람마다 ID를 부여한 뒤 선택한 사람을 추적하는 프로젝트입니다.

현재 코드는 `video_input/input2.mp4`를 기본 테스트 영상으로 사용합니다.

## 주요 기능

- YOLO11 기반 사람 탐지
- 사람 객체별 트래킹 ID 부여
- 마우스 클릭으로 특정 사람 ID 선택
- 선택한 사람 강조 표시 및 추적
- 가려졌다가 다시 나타나는 사람의 ID를 유지하기 위한 stable ID 재할당
- 결과 영상 저장 옵션 지원

## 설치

```powershell
pip install -r requirements.txt
```

## 실행

```powershell
python start.py
```

## 조작 방법

- 사람 박스 클릭: 해당 사람 ID 선택
- `c`: 선택 해제
- `space`: 일시정지/재생
- `q` 또는 `esc`: 종료

## 기본 설정

기본 실행은 탐지 성능을 우선하도록 설정되어 있습니다.

- 모델: `yolo11s.pt`
- 입력 이미지 크기: `1280`
- confidence threshold: `0.20`
- tracker: `botsort.yaml`

`yolo11n.pt`는 빠르지만 드론 시점의 작은 사람을 놓칠 수 있어 기본값은 `yolo11s.pt`로 설정했습니다.

## 인식률 개선 실행 예시

작게 보이는 사람을 더 잘 잡고 싶을 때:

```powershell
python start.py --scale 1.5 --conf 0.15 --imgsz 1536 --enhance
```

빠르게 테스트하고 싶을 때:

```powershell
python start.py --model yolo11n.pt --imgsz 960 --conf 0.25 --tracker bytetrack.yaml
```

결과 영상을 저장하고 싶을 때:

```powershell
python start.py --output output/tracked_input2.mp4
```

처음 20프레임만 빠르게 확인하고 싶을 때:

```powershell
python start.py --model yolo11n.pt --no-display --max-frames 20
```

## 가림 상황에서 ID 유지

기본 트래커는 사람이 가려졌다가 다시 나타나면 새로운 ID를 부여할 수 있습니다. 이를 완화하기 위해 이 프로젝트는 raw tracker ID 위에 stable ID 레이어를 추가했습니다.

화면에 표시되는 `ID`는 stable ID입니다. 내부 트래커가 부여한 원래 ID도 함께 보고 싶으면 다음 옵션을 사용합니다.

```powershell
python start.py --show-raw-id
```

사람이 더 오래 가려지는 상황에서 기존 ID를 더 오래 기억하고 싶을 때:

```powershell
python start.py --reid-missing 180 --reid-threshold 0.38 --show-raw-id
```

사람이 많은 장면에서 서로 다른 사람이 같은 ID로 합쳐지면 threshold를 높여 더 엄격하게 매칭합니다.

```powershell
python start.py --reid-threshold 0.55
```

## 프로젝트 구조

```text
.
├── start.py              # 실행 진입점
├── requirements.txt      # 필요한 Python 패키지
├── src/
│   ├── app.py            # 전체 실행 흐름 및 CLI 옵션
│   ├── config.py         # 기본 설정값
│   ├── detection.py      # YOLO 탐지 및 트래킹 호출
│   ├── models.py         # 데이터 구조
│   ├── reid.py           # stable ID 재할당 로직
│   ├── ui.py             # 화면 표시 및 마우스 선택
│   └── video_io.py       # 영상 입출력
└── video_input/
    └── input2.mp4        # 로컬 테스트 영상
```

## 참고

현재 코드는 pretrained YOLO 모델을 사용합니다. 드론 하향 시점 영상에서 인식률을 더 크게 올리려면 실제 드론 영상 데이터로 YOLO를 fine-tuning 하는 것이 가장 효과적입니다.
