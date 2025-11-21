# -*- coding: utf-8 -*-
# auto_drive_tiki_debug.py
# 기존 로직 그대로 + 디버그 4창(원본/BEV/Binary/SlidingWindow)

import cv2
import numpy as np
import time
import os

from math import *
from tiki.mini import TikiMini
from aruco_tag_action_tiki_final import ArucoTrigger
from yolo_inference import YOLOONNX
from dashboard_comm import DashboardComm
from fire_sector_reader import FireSectorReader
from qr_detector import QRDetector

# 디버그 화면 표시 여부 (SSH 환경에서는 False로 설정)
SHOW_DEBUG_WINDOWS = False


# =========================================================
# BEV (Bird-Eye View) 변환
# =========================================================
def warpping(image):
    # src = np.float32([[140, 460], [0, 840], [1140, 460], [1280, 840]])
    src = np.float32([[70, 230], [0, 420], [570, 230], [640, 420]])
    dst = np.float32([[0, 0], [0, 480], [480, 0], [480, 480]])

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, (480, 480))
    return warp


# =========================================================
# 차선 필터링 (검은 차선만 남기기)
# =========================================================
def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])
    mask = cv2.inRange(hls, black_lower, black_upper)
    return cv2.bitwise_and(image, image, mask=mask)


# =========================================================
# Lane Follower
# =========================================================
class LaneFollower:

    def __init__(self, tiki, aruco_trigger):

        self.tiki = tiki
        self.aruco = aruco_trigger

        self.forward_rpm = 20
        self.rotate_rpm = 10

        self.lat_weight = 1.2
        self.heading_weight = 0.7

        # 디버그 이미지 저장용
        self.dbg_bev = None
        self.dbg_bin = None
        self.dbg_slide = None

    # ------------------------------------------
    def high_level_detect(self, hough_img):

        nwindows = 15
        margin = 90
        minpix = 15

        histogram = np.sum(hough_img[hough_img.shape[0] // 2:, :], axis=0)
        midx_current = np.argmax(histogram)

        window_height = int(hough_img.shape[0] / nwindows)

        nz = hough_img.nonzero()

        x, y = [], []
        mid_sum = 0
        total_loop = 0

        for window in range(nwindows - 4):

            wy_low = hough_img.shape[0] - (window + 1) * window_height
            wy_high = hough_img.shape[0] - window * window_height

            wx_low = midx_current - margin
            wx_high = midx_current + margin

            good = (
                (nz[0] >= wy_low) & (nz[0] < wy_high) &
                (nz[1] >= wx_low) & (nz[1] < wx_high)
            ).nonzero()[0]

            if len(good) > minpix:
                midx_current = int(np.mean(nz[1][good]))

            x.append(midx_current)
            y.append((wy_low + wy_high) / 2)

            mid_sum += midx_current
            total_loop += 1

        if total_loop == 0:
            return np.array([0,0,0]), 0

        fit = np.polyfit(np.array(y[1:]), np.array(x[1:]), 2)
        mid_avg = mid_sum / total_loop
        return fit, mid_avg

    # ------------------------------------------
    def follow(self, frame):

        if self.aruco.mode != "LANE_FOLLOW":
            self.tiki.stop()
            return

        # -------- BEV --------
        bev = warpping(frame)
        self.dbg_bev = bev.copy()
        
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        
        # -------- 필터 --------
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        self.dbg_bin = bin_img.copy()

        # -------- Canny --------
        canny_img = cv2.Canny(bin_img, 10, 100)

        # -------- Hough --------
        lines = cv2.HoughLines(canny_img, 1, np.pi/180, 80)
        
        hough_img = np.zeros((480, 480), dtype=np.uint8)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta); b = np.sin(theta)
                x0 = a*rho; y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                slope = 90 - degrees(atan(b/a))

                if abs(slope) < 5:
                    cv2.line(hough_img, (x1,y1), (x2,y2), 0, 30)
                else:
                    cv2.line(hough_img, (x1,y1), (x2,y2), 255, 8)

        # -------- Sliding Window --------
        fit, avg = self.high_level_detect(hough_img)
        self.dbg_slide = hough_img.copy()

        # -------- PID 조향 --------
        poly = np.poly1d(fit)
        heading = atan(poly[1])

        if fit[0]==0 and fit[1]==0:
            self.tiki.stop()
            return
            
        distance = -(np.polyval(fit, 480) - 240)

        lat_norm = distance / 240.0
        total_err = self.lat_weight * lat_norm + self.heading_weight * heading

        threshold = 0.05

        if total_err > threshold:
            left = self.forward_rpm - self.rotate_rpm
            right = self.forward_rpm + self.rotate_rpm
        elif total_err < -threshold:
            left = self.forward_rpm + self.rotate_rpm
            right = self.forward_rpm - self.rotate_rpm
        else:
            left = right = self.forward_rpm

        self.tiki.set_motor_power(self.tiki.MOTOR_LEFT, int(left))
        self.tiki.set_motor_power(self.tiki.MOTOR_RIGHT, int(right))

        # if fit[0]==0 and fit[1]==0:
        #     self.tiki.set_motor_power(self.tiki.MOTOR_LEFT, int(left))
        #     self.tiki.set_motor_power(self.tiki.MOTOR_RIGHT, int(right))
        #     return


# =========================================================
# Main
# =========================================================
def main():

    tiki = TikiMini()
    tiki.set_motor_mode(tiki.MOTOR_MODE_PID)
    tiki.stop()

    # YOLO 모델 초기화
    yolo_model = None
    yolo_path = "yolo/best.onnx"
    if os.path.exists(yolo_path):
        try:
            yolo_model = YOLOONNX(
                onnx_path=yolo_path,
                conf_threshold=0.25,
                iou_threshold=0.45
            )
            print("[MAIN] YOLO 모델 로드 완료")
        except Exception as e:
            print(f"[MAIN] YOLO 모델 로드 실패: {e}")
            yolo_model = None
    else:
        print(f"[MAIN] YOLO 모델 파일을 찾을 수 없습니다: {yolo_path}")

    # 대시보드 통신 초기화
    dashboard = None
    try:
        dashboard = DashboardComm(
            server_url="http://58.229.150.23:5000",
            mission_code="2W9G"
        )
        print("[MAIN] 대시보드 통신 초기화 완료")
    except Exception as e:
        print(f"[MAIN] 대시보드 통신 초기화 실패: {e}")
        dashboard = None

    # 화재 섹터 리더 초기화
    fire_sector_reader = None
    try:
        fire_sector_reader = FireSectorReader(
            report_dir="report",
            json_filename="2W9G.json",
            poll_interval=1.0
        )
        print("[MAIN] 화재 섹터 리더 초기화 완료")
        
        # JSON 파일이 생성될 때까지 대기
        print("[MAIN] report/2W9G.json 파일을 기다리는 중...")
        if fire_sector_reader.wait_for_file():
            fire_sector_reader.read_fire_sectors()
            print("[MAIN] 화재 섹터 정보 로드 완료 - 주행 시작")
        else:
            print("[MAIN] JSON 파일 대기 시간 초과 - 주행 시작 (화재 섹터 정보 없음)")
    except Exception as e:
        print(f"[MAIN] 화재 섹터 리더 초기화 실패: {e}")
        fire_sector_reader = None

    # QR 코드 감지기 초기화
    qr_detector = None
    try:
        qr_detector = QRDetector()
        if qr_detector.qr_detector is None:
            print("[MAIN] 경고: QR 코드 감지기가 None입니다. OpenCV 버전을 확인하세요.")
            import cv2
            print(f"[MAIN] OpenCV 버전: {cv2.__version__}")
            print(f"[MAIN] cv2.QRCodeDetector 존재 여부: {hasattr(cv2, 'QRCodeDetector')}")
        else:
            print("[MAIN] QR 코드 감지기 초기화 완료")
            # 테스트 감지기 동작 확인
            try:
                import numpy as np
                test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                result = qr_detector.detect(test_frame)
                print(f"[MAIN] QR 감지기 테스트: 정상 (결과 타입: {type(result)})")
            except Exception as e:
                print(f"[MAIN] QR 감지기 테스트 실패: {e}")
    except Exception as e:
        print(f"[MAIN] QR 코드 감지기 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        qr_detector = None

    # ArUco 트리거 초기화 (YOLO, Dashboard, FireSectorReader 전달)
    aruco = ArucoTrigger(
        tiki, 
        straight_rpm=20, 
        turn_rpm=40,
        yolo_model=yolo_model,
        dashboard=dashboard,
        fire_sector_reader=fire_sector_reader
    )
    follower = LaneFollower(tiki, aruco)
    
    # QR 코드 감지 상태
    frame_count = 0
    qr_detection_interval = 1  # 매 프레임마다 실행 (QR 코드 인식 개선)
    last_qr_data = None

    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, -1)
            frame_count += 1

            # 화재 섹터 정보 주기적 업데이트 (매 프레임마다는 아니고 주기적으로)
            if fire_sector_reader and int(time.time()) % 5 == 0:  # 5초마다 한 번씩
                fire_sector_reader.update_fire_sectors()

            # QR 코드 인식 (항상 실행, 매 프레임마다)
            if qr_detector is None:
                if frame_count % 100 == 0:  # 100프레임마다 한 번씩만 로그 출력
                    print("[QR] QR 감지기가 초기화되지 않았습니다.")
            elif frame_count % qr_detection_interval == 0:
                if frame is None:
                    if frame_count % 100 == 0:  # 100프레임마다 한 번씩만 로그 출력
                        print("[QR] 프레임이 None입니다.")
                else:
                    # 디버그: 주기적으로 QR 감지 시도 로그
                    if frame_count % 300 == 0:  # 10초마다 (30fps 기준)
                        print(f"[QR] QR 감지 시도 중... (프레임: {frame_count}, 크기: {frame.shape})")
                    
                    qr_codes = qr_detector.detect(frame)
                    
                    if qr_codes:
                        for qr in qr_codes:
                            qr_data = qr['data']
                            # 새로운 QR 코드 감지 시 LED 제어 및 대시보드 전송
                            if qr_data != last_qr_data:
                                last_qr_data = qr_data
                                print(f"[QR] QR 코드 감지: {qr_data}")
                                
                                # QR 코드 데이터 파싱 및 LED 제어
                                try:
                                    if qr_data.startswith("ID_"):
                                        parts = qr_data.split("_")
                                        if len(parts) >= 3:
                                            shape_str = parts[1]  # O, X, #
                                            color_str = parts[2].upper()  # R, G, B
                                            
                                            # LED 색상 값 설정 (RGB 형식)
                                            if color_str == 'R':
                                                r, g, b = 50, 0, 0  # 빨강
                                                led_color_name = "red"
                                            elif color_str == 'G':
                                                r, g, b = 0, 50, 0  # 녹색
                                                led_color_name = "green"
                                            elif color_str == 'B':
                                                r, g, b = 0, 0, 50  # 파랑
                                                led_color_name = "blue"
                                            else:
                                                r, g, b = 0, 50, 0  # 기본: 녹색
                                                led_color_name = "green"
                                            
                                            # Tiki LED 제어 (top 16-bit LED Strip 전체)
                                            try:
                                                # 모든 LED에 같은 색상 설정
                                                for i in range(16):
                                                    tiki.set_led(0, i, r, g, b)
                                                print(f"[QR] LED 켜짐: {led_color_name} (모양: {shape_str})")
                                            except Exception as e:
                                                print(f"[QR] LED 제어 오류: {e}")
                                                
                                except Exception as e:
                                    print(f"[QR] QR 데이터 파싱 오류: {e}")
                                
                                # 대시보드로 포인트 추가
                                if dashboard:
                                    point_name = f"qr_{qr_data[:10]}"  # QR 데이터의 처음 10자만 사용
                                    dashboard.add_point(point_name)
                    elif frame_count % 300 == 0:  # 10초마다 (30fps 기준)
                        print(f"[QR] QR 코드 미감지 (프레임: {frame_count})")

            # Aruco
            aruco.observe_and_maybe_trigger(frame)

            # Lane Follow
            follower.follow(frame)
            
            # if action, action execute
            aruco.step()

            # ===== 디버그 화면 출력 (GUI 환경에서만) =====
            if SHOW_DEBUG_WINDOWS:
                cv2.imshow("Original", frame)
                if follower.dbg_bev is not None:
                    cv2.imshow("BEV", follower.dbg_bev)
                if follower.dbg_bin is not None:
                    cv2.imshow("Binary", follower.dbg_bin)
                if follower.dbg_slide is not None:
                    cv2.imshow("Sliding Window", follower.dbg_slide)
                
                # ESC 키로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
            # ==============================

    except KeyboardInterrupt:
        print("STOP")

    finally:
        tiki.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
