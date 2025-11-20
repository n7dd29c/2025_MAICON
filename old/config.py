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
MAX_SPEED = 0.6  # 0.0 ~ 1.0 (0.5 -> 0.6, 속도 향상)
MIN_SPEED = 0.3  # 0.0 ~ 1.0 (0.2 -> 0.3, 최소 속도 향상)
SAFE_OFFSET_THRESHOLD = 40  # 픽셀 (50 -> 40, 더 엄격한 기준)
STEERING_DEADZONE = 5  # 픽셀 (3 -> 5, 작은 오프셋 무시로 비틀거림 방지)
CENTER_LANE_KP_MULTIPLIER = 1.2  # 중앙선 감지 시 KP 배율 (1.5 -> 1.2, 과도한 반응 방지)
STRAIGHT_DRIVING_THRESHOLD = 15  # 픽셀 (이 값 이하면 직진 모드)
STRAIGHT_DRIVING_SMOOTHING = 0.85  # 직진 모드 스무딩 계수 (높을수록 부드러움)

# 곡선 주행 설정
USE_CURVE_FITTING = True  # 곡선 피팅 사용 여부 (2차 다항식)
CURVE_DETECTION_THRESHOLD = 0.008  # 곡선 감지 임계값 (0.005 -> 0.008, 직선 오인 방지)
CURVE_STEERING_MULTIPLIER = 1.5  # 곡선 구간 조향 배율 (증가)

# 성능 설정
SMOOTHING_FACTOR = 0.7  # 차선 스무딩 계수
FRAME_SKIP = 1  # 프레임 스킵 (홀수/짝수 분리)

# Bird's Eye View 설정
USE_BIRD_VIEW = True  # Bird's Eye View 사용 여부
BIRD_VIEW_TOP_WIDTH_RATIO = 0.6  # 상단 너비 비율 (0.6 -> 0.8, 카메라가 아래를 향하므로 더 넓게)
BIRD_VIEW_BOTTOM_MARGIN_RATIO = 0.2  # 하단 여유 비율 (양쪽 10%)
BIRD_VIEW_TOP_OFFSET_RATIO = 0.05  # 상단 포인트 오프셋 (상단에서 5% 아래로, 0 -> 0.05)

# 포트홀 감지 설정
USE_POTHOLE_AVOIDANCE = False  # 포트홀 회피 사용 여부 (False: 회피 안 함, 통신 보고만)
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
YOLO_MODEL_PATH = "yolo/Object7.v7-maicon_mortar_background.yolov8/yolov8n18/weights/best.onnx"  # ONNX 모델 경로
YOLO_CONF_THRESHOLD = 0.25  # 신뢰도 임계값
YOLO_IOU_THRESHOLD = 0.45  # NMS IoU 임계값
YOLO_CLASS_NAMES = ["Hazmat", "Missile", "Enemy", "Tank", "Car", "Mortar", "Box"]  # TODO: 실제 학습된 클래스 목록으로 변경
YOLO_AVOIDANCE_CLASSES = []  # 회피가 필요한 객체 클래스
YOLO_DISPLAY_ONLY_CLASSES = []  # 표시만 하는 객체 클래스

# ArUco 마커 설정 (베이스라인 코드 기준)
USE_ARUCO = True  # ArUco 마커 사용 여부
ARUCO_DICTIONARY = 10  # cv2.aruco.DICT_4X4_50 (10) - 베이스라인 코드 기준
ARUCO_CONTROL_ENABLED = False  # ArUco 마커로 주행 제어 여부 (False: 통신 보고만, True: 주행 제어도)
ARUCO_TURN_ANGLE = 25.0  # 회전 각도 (도)
ARUCO_TURN_DURATION = 2.0  # 회전 지속 시간 (초)
ARUCO_TURN_SPEED = 0.4  # 회전 시 속도 (0.0 ~ 1.0)

# ArUco 마커 ID별 동작 설정 (화재 건물 마커만 회전 설정)
# 화재 건물이 아닌 마커는 'go_straight'로 설정 (행동 제어 안 함)
ARUCO_MARKER_ACTIONS = {
    # 화재 건물 섹터 마커들 (회전 방향 설정)
    2: 'turn_left',    # ID 2 (sector1) → 좌회전
    3: 'turn_right',   # ID 3 (sector2) → 우회전
    4: 'turn_left',    # ID 4 (sector3) → 좌회전
    5: 'turn_right',   # ID 5 (sector4) → 우회전
    6: 'turn_left',    # ID 6 (sector5) → 좌회전
    7: 'turn_right',   # ID 7 (sector6) → 우회전
    9: 'turn_left',    # ID 9 (sector7) → 좌회전
    11: 'turn_right',  # ID 11 (sector8) → 우회전
    12: 'turn_left',   # ID 12 (sector9) → 좌회전
    # 화재 건물이 아닌 마커들 (행동 제어 안 함)
    1: 'go_straight',  # ID 1 (Alpha) → 직진 (차선 추종)
    8: 'go_straight',  # ID 8 (Bravo) → 직진 (차선 추종)
    10: 'go_straight', # ID 10 (Charlie) → 직진 (차선 추종)
    13: 'go_straight', # ID 13 (Finish) → 직진 (차선 추종)
    # 기본값: 설정되지 않은 마커는 직진
}

# 화재 건물 마커 설정
FIRE_BUILDING_TURN_ANGLE = 45.0  # 화재 건물 마커 감지 시 회전 각도 (도)
FIRE_BUILDING_TURN_DURATION = 3.0  # 회전 지속 시간 (초)
FIRE_BUILDING_CAPTURE_DELAY = 0.5  # 회전 완료 후 캡쳐 대기 시간 (초)
FIRE_BUILDING_RETURN_DURATION = 3.0  # 원상복구 회전 지속 시간 (초)
FIRE_BUILDING_CAPTURE_DIR = "fire_captures"  # 화재 건물 사진 저장 디렉터리

# ArUco 마커 ID별 섹터 설정 (대시보드 통신용)
# 마커 ID별로 다른 섹터를 설정할 수 있음
# 설정되지 않은 마커는 DASHBOARD_SECTOR 사용
ARUCO_MARKER_SECTORS = {
    # 예시: 마커 ID별 섹터 설정
    1: "Alpha",     # ID 1 → Alpha 섹터
    2: "sector1",   # ID 2 → sector1
    3: "sector2",   # ID 3 → sector2
    4: "sector3",   # ID 4 → sector3
    5: "sector4",   # ID 5 → sector4
    6: "sector5",   # ID 6 → sector5
    7: "sector6",   # ID 7 → sector6
    8: "Bravo",     # ID 8 → Bravo 섹터
    9: "sector7",   # ID 9 → sector7
    10: "Charlie",  # ID 10 → Charlie 섹터
    11: "sector8",  # ID 11 → sector8
    12: "sector9",  # ID 12 → sector9
    13: "Finish",   # ID 13 → 임무종료
    # ... (필요한 마커 ID만 설정)
}

# QR 코드 설정
USE_QR = True  # QR 코드 인식 사용 여부
QR_LED_DURATION = 3.0  # LED 점등 지속 시간 (초)
# QR 코드 데이터 형식: ID_SHAPE_COLOR (예: ID_O_R, ID_X_G, ID_#_B)
# SHAPE: O(원형), X(X모양), #(격자), R(사각형)
# COLOR: R(빨강), G(녹색), B(파랑) - LED 색상도 자동으로 매핑됨

# 대시보드 통신 설정
USE_DASHBOARD = True  # 대시보드 통신 사용 여부
DASHBOARD_SERVER_URL = "http://58.229.150.23:5000"  # 대시보드 서버 URL
MISSION_CODE = "2W9G"  # 임무 코드
DASHBOARD_SECTOR = "Alpha"  # 현재 로봇 섹터 ("Alpha", "Bravo", "Charlie")
DASHBOARD_UPDATE_INTERVAL = 3.0  # 대시보드 업데이트 간격 (초)

# 화재 섹터 정보 파일 경로
FIRE_SECTOR_REPORT_DIR = "report"  # 드론이 생성한 JSON 파일이 있는 디렉터리
FIRE_SECTOR_JSON_FILENAME = f"{MISSION_CODE}.json"  # 화재 섹터 정보 JSON 파일명

