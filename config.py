"""
설정 파일
Jetson Nano 및 환경에 맞게 조정 가능
"""

# 카메라 설정
CAMERA_ID = 0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FPS_TARGET = 15
USE_GSTREAMER = True  # Jetson Nano CSI 카메라 사용 시 True

# 차선 인식 설정
ROI_RATIO = 1.0  # ROI 비율 (카메라가 아래를 향하므로 전체 사용)
WHITE_LANE_LOWER = [0, 0, 200]  # HSV
WHITE_LANE_UPPER = [180, 30, 255]
YELLOW_LANE_LOWER = [20, 100, 100]  # HSV
YELLOW_LANE_UPPER = [30, 255, 255]

# 제어 설정
MAX_STEERING_ANGLE = 25.0  # 도 (30 -> 25, 과도한 조향 방지)
KP = 0.5  # 비례 게인 (0.3 -> 0.5, 중앙선 추종 강화)
KD = 0.2  # 미분 게인 (0.15 -> 0.2, 안정성 향상)
MAX_SPEED = 0.5  # 0.0 ~ 1.0
MIN_SPEED = 0.2  # 0.0 ~ 1.0
SAFE_OFFSET_THRESHOLD = 40  # 픽셀 (50 -> 40, 더 엄격한 기준)
STEERING_DEADZONE = 3  # 픽셀 (5 -> 3, 중앙선 추종 민감도 향상)
CENTER_LANE_KP_MULTIPLIER = 1.5  # 중앙선 감지 시 KP 배율 (더 강한 반응)

# 곡선 주행 설정
USE_CURVE_FITTING = True  # 곡선 피팅 사용 여부 (2차 다항식)
CURVE_DETECTION_THRESHOLD = 0.01  # 곡선 감지 임계값 (2차 계수 절댓값)
CURVE_STEERING_MULTIPLIER = 1.2  # 곡선 구간 조향 배율

# 성능 설정
SMOOTHING_FACTOR = 0.7  # 차선 스무딩 계수
FRAME_SKIP = 1  # 프레임 스킵 (홀수/짝수 분리)

# Bird's Eye View 설정
USE_BIRD_VIEW = True  # Bird's Eye View 사용 여부
BIRD_VIEW_TOP_WIDTH_RATIO = 0.6  # 상단 너비 비율 (0.6 -> 0.8, 카메라가 아래를 향하므로 더 넓게)
BIRD_VIEW_BOTTOM_MARGIN_RATIO = 0.2  # 하단 여유 비율 (양쪽 10%)
BIRD_VIEW_TOP_OFFSET_RATIO = 0.05  # 상단 포인트 오프셋 (상단에서 5% 아래로, 0 -> 0.05)

# 포트홀 감지 설정
POTHOLE_MIN_AREA = 100  # 최소 포트홀 면적 (픽셀)
POTHOLE_MAX_AREA = 50000  # 최대 포트홀 면적 (픽셀)
POTHOLE_AVOIDANCE_MARGIN = 50  # 회피 안전 거리 (픽셀)
POTHOLE_RETURN_THRESHOLD = 30  # 복귀 임계값 (픽셀)

# 하드웨어 설정
USE_TIKI = True  # TikiMini API 사용 여부
MOTOR_MODE = "PID"  # "PWM" 또는 "PID"
TURN_SENSITIVITY = 0.5  # 조향 민감도 (0.0 ~ 1.0)

# YOLO 객체 탐지 설정
USE_YOLO = True  # YOLO 사용 여부 (False면 전통적인 CV 방법 사용)
YOLO_ENGINE_PATH = "yolo/Object7.v7-maicon_mortar_background.yolov8/yolov8n/weights/best.trt"  # TensorRT 엔진 경로
YOLO_CONF_THRESHOLD = 0.25  # 신뢰도 임계값
YOLO_IOU_THRESHOLD = 0.45  # NMS IoU 임계값
YOLO_CLASS_NAMES = ["unknown"]  # TODO: 실제 학습된 클래스 목록으로 변경
YOLO_AVOIDANCE_CLASSES = []  # 회피가 필요한 객체 클래스
YOLO_DISPLAY_ONLY_CLASSES = []  # 표시만 하는 객체 클래스

# ArUco 마커 설정
USE_ARUCO = False  # ArUco 마커 사용 여부
ARUCO_DICTIONARY = 8  # cv2.aruco.DICT_6X6_50 (8) - 회전된 마커 감지 개선
ARUCO_TURN_ANGLE = 25.0  # 회전 각도 (도)
ARUCO_TURN_DURATION = 2.0  # 회전 지속 시간 (초)
ARUCO_TURN_SPEED = 0.4  # 회전 시 속도 (0.0 ~ 1.0)

# QR 코드 설정 (추후 구현)
USE_QR = False  # QR 코드 인식 사용 여부

