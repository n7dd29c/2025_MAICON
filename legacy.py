# -*- coding: utf-8 -*-
# auto_drive_tiki_debug_ver6.py
# - HoughLines 제거
# - Binary + Sliding Window로 곡선 포함 중앙선 추적
# - 직각 교차로는 "직진 모드"로 통과
# - 디버그 4창(원본 / BEV / Binary / Sliding) 유지
import os 
import cv2
import numpy as np
import time
import ipywidgets as widgets
from math import atan

from tiki.mini import TikiMini
from aruco_tag_action_tiki_final import ArucoTrigger
from IPython.display import display


# =========================================================
# 디버그 UI 4개
# =========================================================
w1 = widgets.Image(format='jpeg')   # 원본
w2 = widgets.Image(format='jpeg')   # BEV
w3 = widgets.Image(format='jpeg')   # Binary
w4 = widgets.Image(format='jpeg')   # Sliding Window 결과

display(w1, w2, w3, w4)


def to_bytes(img):
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    return buf.tobytes()


# =========================================================
# BEV 변환 (필요하면 src만 조금씩 조정해서 쓰면 됨)
# =========================================================
def warpping(image):
    src = np.float32([
        [70, 230],   # 좌상
        [0,  420],   # 좌하
        [570, 230],  # 우상
        [640, 420]   # 우하
    ])
    dst = np.float32([
        [0, 0],
        [0, 480],
        [480, 0],
        [480, 480]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, (480, 480))
    return warp


# =========================================================
# 흰색 중앙선 필터
# =========================================================
def white_lane_filter(bev_bgr):
    hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)

    # 흰색 계열만 (중앙선) 추출
    white_low  = np.array([0, 170, 0])
    white_high = np.array([180, 255, 255])

    mask = cv2.inRange(hls, white_low, white_high)

    # 노이즈 제거/선 연결
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


# =========================================================
# Sliding Window
#   - 직선/곡선 상관없이 중앙선 전체를 따라가면서 x 좌표 리스트(xs) 뽑기
# =========================================================
def sliding_window(binary_img):
    """
    binary_img: 0/255, shape(H, W)
    return: fit (2차 poly), xs, ys, out_img
    """
    h, w = binary_img.shape

    # 좌우 너무 바깥쪽 라인/벽 제거 (중앙선 근처만 사용)
    binary = binary_img.copy()
    binary[:, :80] = 0
    binary[:, 400:] = 0

    nonzero = binary.nonzero()
    total_pix = len(nonzero[0])

    histogram = np.sum(binary[h // 2:, :], axis=0)
    if np.max(histogram) == 0 or total_pix < 50:
        out = np.dstack((binary, binary, binary))
        return None, [], [], out

    x_current = np.argmax(histogram)

    nwindows = 18
    margin = 60
    minpix = 20
    window_height = h // nwindows

    out = np.dstack((binary, binary, binary))
    xs, ys = [], []

    for win in range(nwindows):
        wy_low = h - (win + 1) * window_height
        wy_high = h - win * window_height

        wx_low = x_current - margin
        wx_high = x_current + margin

        cv2.rectangle(out, (wx_low, wy_low), (wx_high, wy_high),
                      (0, 255, 0), 2)

        good = (
            (nonzero[0] >= wy_low) & (nonzero[0] < wy_high) &
            (nonzero[1] >= wx_low) & (nonzero[1] < wx_high)
        ).nonzero()[0]

        if len(good) > minpix:
            x_current = int(np.mean(nonzero[1][good]))
            out[nonzero[0][good], nonzero[1][good]] = [255, 0, 0]

        xs.append(x_current)
        ys.append((wy_low + wy_high) / 2)

    if len(xs) < 4:
        return None, xs, ys, out

    fit = np.polyfit(ys, xs, 2)

    # 중앙 곡선 시각화 (초기값은 노란색)
    yy = np.linspace(0, h-1, h)
    xx = np.polyval(fit, yy)
    for y, x in zip(yy.astype(int), xx.astype(int)):
        if 0 <= x < w:
            out[y, x] = (0, 255, 255)  # yellow

    return fit, xs, ys, out


# =========================================================
# Lane Follower (곡선 + 직각 교차로 분리 처리)
# =========================================================
class LaneFollower:

    def __init__(self, tiki, aruco_trigger):

        self.tiki = tiki
        self.aruco = aruco_trigger

        self.forward_rpm = 40      # 기본 속도
        self.rotate_gain = 25      # 조향 강도

        self.lat_weight = 1.5      # 중앙에서 벗어난 정도
        self.heading_weight = 0.4  # 각도 기울기

        # 디버그용
        self.dbg_bev = None
        self.dbg_bin = None
        self.dbg_slide = None

        # 교차로 판정용
        self.prev_xs = None
        self.intersection_countdown = 0  # >0 이면 직진 유지 모드

    # ------------------------------------------
    def detect_intersection(self, xs):
        """
        xs: sliding window에서 얻은 각 윈도우의 중앙 x 좌표 리스트
        직각 교차로처럼 라인이 갑자기 꺾이거나 여러 선이 섞인 경우
        → 연속성이 깨지므로 그걸 이용해서 판단
        """
        if len(xs) < 6:
            return False

        xs = np.array(xs, dtype=np.float32)

        # 아래쪽(로봇 가까운 쪽) 6개 윈도우만 사용
        tail = xs[:6]

        # 인접한 차이
        dx = np.diff(tail)
        dx_abs = np.abs(dx)

        max_jump = np.max(dx_abs)
        mean_jump = np.mean(dx_abs)

        # 직각 교차로일 때: 아래쪽에서 x가 갑자기 크게 꺾이는 패턴
        # 곡선: dx가 부드럽게, max_jump/mean_jump 둘 다 상대적으로 작음
        if max_jump > 80 and mean_jump > 25:
            return True

        # 이전 프레임과 비교해도 갑자기 많이 튀면 교차로로 본다
        if self.prev_xs is not None:
            prev_tail = np.array(self.prev_xs[:6], dtype=np.float32)
            if len(prev_tail) == len(tail):
                diff_prev = np.abs(tail - prev_tail)
                if np.max(diff_prev) > 90:
                    return True

        return False

    # ------------------------------------------
    def follow(self, frame):

        #if self.aruco.mode != "LANE_FOLLOW":
        #    self.tiki.stop()
        #    return

        # ---- 1) BEV ----
        bev = warpping(frame)
        self.dbg_bev = bev.copy()

        # ---- 2) 흰색 차선 필터 → Binary ----
        mask = white_lane_filter(bev)
        self.dbg_bin = mask.copy()

        # ---- 3) Sliding Window ----
        fit, xs, ys, slide_img = sliding_window(mask)
        self.dbg_slide = slide_img
        self.prev_xs = xs

        if fit is None:
            # 아무것도 못찾으면 잠깐 멈추고 다음 프레임
            #self.tiki.stop()
            return

        # ---- 4) 교차로 탐지 ----
        is_intersection = self.detect_intersection(xs)

        # 교차로를 새로 발견하면, n프레임 동안 직진 유지
        if is_intersection:
            self.intersection_countdown = 10  # 대략 10프레임 직진

        # ---- 5) 교차로 직진 모드 ----
        if self.intersection_countdown > 0:
            self.intersection_countdown -= 1

            # 중앙선은 파란색으로 그려서 디버깅 (교차로 모드 표시)
            h, w, _ = self.dbg_slide.shape
            poly = np.poly1d(fit)
            yy = np.linspace(0, h-1, h)
            xx = np.polyval(fit, yy)
            for y, x in zip(yy.astype(int), xx.astype(int)):
                if 0 <= x < w:
                    self.dbg_slide[y, x] = (255, 0, 0)  # 파란색

            # 조향 없이 직진
            base = 40  # 살짝 속도 줄이고 직진
            self.tiki.set_motor_power(self.tiki.MOTOR_LEFT, base)
            self.tiki.set_motor_power(self.tiki.MOTOR_RIGHT, base)
            return

        # ---- 6) 일반 곡선/직선 구간 조향 ----
        poly = np.poly1d(fit)
        heading = atan(poly[1])

        # 맨 아래 쪽에서 중앙과의 거리
        lane_x_bottom = np.polyval(fit, 470)
        distance = -(lane_x_bottom - 240.0)   # +면 왼쪽, -면 오른쪽
        lat_norm = distance / 240.0

        # 곡률에 따라 속도 조절 (강한 커브에서 속도 살짝 줄이기)
        curve_strength = abs(poly[1])
        if curve_strength > 0.003:
            base = 20
        else:
            base = self.forward_rpm

        total_err = self.lat_weight * lat_norm + self.heading_weight * heading
        steer = np.clip(total_err, -1.0, 1.0)

        delta = int(abs(steer) * self.rotate_gain)

        if steer > 0.05:
            left = base - delta
            right = base + delta
        elif steer < -0.05:
            left = base + delta
            right = base - delta
        else:
            left = right = base

        self.tiki.set_motor_power(self.tiki.MOTOR_LEFT, int(left))
        self.tiki.set_motor_power(self.tiki.MOTOR_RIGHT, int(right))


# =========================================================
# Main
# =========================================================
def main():
    tiki = TikiMini()
    tiki.set_motor_mode(tiki.MOTOR_MODE_PID)
    tiki.stop()
    
    aruco = ArucoTrigger(tiki, straight_rpm=40, turn_rpm=40)
    follower = LaneFollower(tiki, aruco)

    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=6/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

    # -------------------------------
    # YOLO 모델 초기화
    # -------------------------------
    yolo_model = None
    yolo_path = "yolo/best.onnx"

    if os.path.exists(yolo_path):
        try:
            yolo_model = YOLOONNX(
                onnx_path=yolo_path,
                conf_threshold=0.4,
                iou_threshold=0.45
            )
            print("[MAIN] YOLO 로드 완료")
        except Exception as e:
            print(f"[MAIN] YOLO 로드 실패 : {e}")
    else:
        print(f"[MAIN] YOLO 파일 없음: {yolo_path}")


    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, -1)

            # ArUco 처리
            aruco.observe_and_maybe_trigger(frame)

            # 차선 추종
            follower.follow(frame)

            # ArUco 액션 step
            aruco.step()

            # 디버그 화면
            w1.value = to_bytes(frame)
            if follower.dbg_bev is not None:
                w2.value = to_bytes(follower.dbg_bev)
            if follower.dbg_bin is not None:
                w3.value = to_bytes(follower.dbg_bin)
            if follower.dbg_slide is not None:
                w4.value = to_bytes(follower.dbg_slide)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("STOP")

    finally:
        tiki.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
