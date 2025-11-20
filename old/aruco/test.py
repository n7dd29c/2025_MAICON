import cv2
import numpy as np

img = cv2.imread("aruco/imgs/1.jpg")
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

print(f"이미지 크기: {img.shape[1]}x{img.shape[0]}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 엣지 검출
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\n감지된 윤곽선 수: {len(contours)}")

# 모든 윤곽선 분석
rectangles = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 50:  # 최소 크기 낮춤
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 사각형인지 확인
        if len(approx) == 4:
            # 정사각형에 가까운지 확인
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            rectangles.append({
                'index': i,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'bbox': (x, y, w, h)
            })
            
            print(f"사각형 #{i}: 면적={area:.0f}, 비율={aspect_ratio:.2f}, 위치=({x},{y}), 크기={w}x{h}")

print(f"\n총 {len(rectangles)}개의 사각형 윤곽선 발견")

# ArUco 마커 감지 시도 (개선된 파라미터)
print("\n" + "="*60)
print("ArUco 마커 감지 시도")
print("="*60)

dictionaries = [
    (10, "DICT_4X4_50"),
    (11, "DICT_4X4_100"),
    (12, "DICT_4X4_250"),
    (13, "DICT_4X4_1000"),
    (7, "DICT_5X5_50"),
    (8, "DICT_6X6_50"),
]

# 여러 전처리 방법 시도
preprocess_methods = [
    ("원본", gray),
    ("대비 향상", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)),
    ("이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
    ("역이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
]

for dict_id, dict_name in dictionaries:
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        aruco_params = cv2.aruco.DetectorParameters()
        
        # 매우 관대한 파라미터
        aruco_params.minMarkerPerimeterRate = 0.01
        aruco_params.maxMarkerPerimeterRate = 20.0
        aruco_params.adaptiveThreshConstant = 3
        aruco_params.errorCorrectionRate = 1.0
        
        for method_name, processed_img in preprocess_methods:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                processed_img, aruco_dict, parameters=aruco_params
            )
            
            if ids is not None and len(ids) > 0:
                print(f"✓ {dict_name} ({method_name}): ID={ids[0][0]}")
                # 마커 표시
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                cv2.imshow(f'Detected: {dict_name}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                exit()
    except Exception as e:
        print(f"오류 ({dict_name}): {e}")

print("\n모든 딕셔너리에서 감지 실패")
print("\n가능한 원인:")
print("1. 마커가 ArUco 마커가 아닐 수 있음 (AprilTag, QR Code 등)")
print("2. 마커가 손상되었거나 불완전함")
print("3. 이미지 품질 문제 (해상도, 블러 등)")