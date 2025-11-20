# Jetson Nano 자율주행 시스템

Jetson Nano 기반 차선 인식 자율주행 시스템입니다.

## 기능

- **차선 인식**: 흰색(중앙) 및 노란색(양쪽) 차선 감지
- **자율주행 제어**: 차선 중심 유지 및 조향/속도 제어
- **프레임 분리**: 홀수 프레임(자율주행), 짝수 프레임(객체 디텍션)
- **Jetson Nano 최적화**: 성능 최적화된 이미지 처리

## 프로젝트 구조!

```
lanedetection/
├── lane_detector.py          # 차선 인식 모듈
├── autonomous_control.py     # 자율주행 제어 모듈
├── autonomous_driving.py     # 메인 실행 파일
├── config.py                 # 설정 파일
├── requirements.txt          # 필수 라이브러리
└── README.md                 # 이 파일
```

## 설치

1. 가상환경 생성 (권장):
```bash
python3 -m venv venv
source venv/bin/activate  # Linux
# 또는
venv\Scripts\activate  # Windows
```

2. 라이브러리 설치:
```bash
pip install -r requirements.txt
```

## 사용법

### 기본 실행
```bash
python autonomous_driving.py
```

### 옵션
```bash
python autonomous_driving.py --camera 0 --width 640 --height 480 --fps 15
```

### 주요 옵션
- `--camera`: 카메라 ID (기본값: 0)
- `--width`: 이미지 너비 (기본값: 640)
- `--height`: 이미지 높이 (기본값: 480)
- `--fps`: 목표 FPS (기본값: 15)
- `--no-preview`: 화면 미리보기 비활성화
- `--save-video`: 처리된 비디오 저장

## 동작 원리

1. **홀수 프레임 처리**: 차선 인식 및 자율주행 제어
2. **차선 감지**: HSV 색상 공간에서 흰색/노란색 차선 추출
3. **Hough 변환**: 직선 검출을 통한 차선 추정
4. **제어 알고리즘**: PID 제어를 통한 조향 및 속도 제어

## 설정 조정

`config.py` 파일에서 다음 설정을 조정할 수 있습니다:

- 카메라 해상도 및 FPS
- 차선 색상 범위 (HSV)
- 제어 게인 (KP, KD)
- 최대/최소 속도
- 안전 임계값

## 하드웨어 연결

실제 모터/서보 제어를 위해서는 `autonomous_control.py`의 `execute_control()` 메서드를 수정하여 하드웨어 인터페이스를 추가해야 합니다.

## 성능 최적화

Jetson Nano에서 최적 성능을 위해:
- 이미지 해상도 조정 (640x480 권장)
- FPS 목표 설정 (15 FPS 권장)
- 프레임 스킵 활용 (홀수/짝수 분리)

## 다음 단계

- 짝수 프레임에서 객체 디텍션 모듈 추가
- ArUco 마커 인식 기능 추가
- 실제 하드웨어 제어 인터페이스 구현

