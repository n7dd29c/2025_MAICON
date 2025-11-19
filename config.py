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
ROI_RATIO = 0.6  # ROI 비율 (하단 부분만 사용)
WHITE_LANE_LOWER = [0, 0, 200]  # HSV
WHITE_LANE_UPPER = [180, 30, 255]
YELLOW_LANE_LOWER = [20, 100, 100]  # HSV
YELLOW_LANE_UPPER = [30, 255, 255]

# 제어 설정
MAX_STEERING_ANGLE = 30.0  # 도
KP = 0.5  # 비례 게인
KD = 0.1  # 미분 게인
MAX_SPEED = 0.5  # 0.0 ~ 1.0
MIN_SPEED = 0.2  # 0.0 ~ 1.0
SAFE_OFFSET_THRESHOLD = 50  # 픽셀

# 성능 설정
SMOOTHING_FACTOR = 0.7  # 차선 스무딩 계수
FRAME_SKIP = 1  # 프레임 스킵 (홀수/짝수 분리)

# Bird's Eye View 설정
USE_BIRD_VIEW = True  # Bird's Eye View 사용 여부

# 포트홀 감지 설정
POTHOLE_MIN_AREA = 100  # 최소 포트홀 면적 (픽셀)
POTHOLE_MAX_AREA = 50000  # 최대 포트홀 면적 (픽셀)
POTHOLE_AVOIDANCE_MARGIN = 50  # 회피 안전 거리 (픽셀)
POTHOLE_RETURN_THRESHOLD = 30  # 복귀 임계값 (픽셀)

# 하드웨어 설정
USE_TIKI = True  # TikiMini API 사용 여부
MOTOR_MODE = "PID"  # "PWM" 또는 "PID"
TURN_SENSITIVITY = 0.5  # 조향 민감도 (0.0 ~ 1.0)

