# -*- coding: utf-8 -*-
# aruco_tag_action_tiki.py (FINAL)

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Optional, List, Dict

# YOLO 클래스 이름 (config.py 기준)
YOLO_CLASS_NAMES = ["Hazmat", "Missile", "Enemy", "Tank", "Car", "Mortar", "Box"]

# ------------------------------------------
# ArUco 기본 세팅
# ------------------------------------------

try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
except AttributeError:
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
except AttributeError:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()


# ------------------------------------------
# ArUco Detector
# ------------------------------------------

class ArucoDetector(object):
    def __init__(self):
        pass

    def detect_ids(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        results = []
        if ids is not None:
            ids = ids.flatten()
            for c, i in zip(corners, ids):
                pts = c.reshape(-1, 2)
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                area = abs(w * h)
                results.append({"id": int(i), "center": (cx, cy), "area": area})
        return results


# ------------------------------------------
# ArUco Trigger
# ------------------------------------------

class ArucoTrigger(object):

    def __init__(self, tiki,
                 straight_rpm=20,      # 직진/후진
                 turn_rpm=40,          # 회전 rpm
                 yolo_model=None,      # YOLO 모델 (YOLOONNX)
                 dashboard=None,       # 대시보드 통신 (DashboardComm)
                 fire_sector_reader=None):  # 화재 섹터 읽기 (FireSectorReader)

        self.tiki = tiki
        self.straight_rpm = straight_rpm
        self.turn_rpm = turn_rpm
        self.yolo_model = yolo_model
        self.dashboard = dashboard
        self.fire_sector_reader = fire_sector_reader

        # 1: Alpha, 8: Bravo, 10: Charlie, 13: Finish
        self.marker_to_sector = {
            1: "Alpha",
            2: "sector1",
            3: "sector2",
            4: "sector3",
            5: "sector4",
            6: "sector5",
            7: "sector6",
            8: "Bravo",
            9: "sector7",
            10: "Charlie",
            11: "sector8",
            12: "sector9",
            13: "Finish"
        }
        self.current_sector = None
        self._last_marker_id = None

        # 사용자가 직접 duration(sec)을 나중에 조정
        self.rules = {
            1: {1: [("right", 1.5), ("YOLO", 0.0), ("left", 1.5), ("CHECKPOINT_PASSING", 0.0)]},
            2: {1: [("right", 0,5), ("YOLO", 0.0)]},
            3: {1: [("YOLO", 0.0)]},
            4: {1: [("YOLO", 0.0)]},
            5: {1: [("YOLO", 0.0)]},
            6: {1: [("YOLO", 0.0), ("right", 0.5), ("YOLO", 0.0)]},
            7: {1: [("left", 1), ("YOLO", 0.0), ("right", 1)]},
            8: {1: [("POTHOLE_AVOIDANCE", 0.0), ("right", 0.5), ("YOLO", 0.0), ("left", 0.5), ("CHECKPOINT_PASSING", 0.0)]},
            9: {1: [("left", 1), ("YOLO", 0.0), ("right", 1)]},
            10: {1:[("CHECKPOINT_PASSING", 0.0), ("POTHOLE_AVOIDANCE")]},
            11: {1:[("left", 0.5), ("YOLO", 0.0), ("right", 0.5)]},
            12: {1:[("left", 0.5), ("YOLO", 0.0), ("right", 0.5)]},
            13: {}
        }

        # ArUco 감지기
        self.detector = ArucoDetector()

        # 상태 변수
        self.mode = "LANE_FOLLOW"
        self.pending_actions = []
        self.seen_counts = {}

        # 쿨다운
        self.cooldown_default = 10.0
        self.cooldown_per_id = {}
        self.last_trigger_times = {}

        # 프레임 필터링
        self.required_consecutive = 2
        self._consec = {}

        self.min_area = 60.0
        self.min_y = 60.0
        self.max_y = 460.0

        # 이미지 저장 경로
        self.image_dir = Path("captured_images")
        self.image_dir.mkdir(exist_ok=True)

    def _gate(self, det):
        # 면적 조건: det["area"]가 최소 면적 이상인지 검사
        area_ok = det["area"] >= self.min_area

        # y좌표를 center에서 가져옴 (center는 (x,y) 튜플)
        y = det["center"][1]

        # y좌표가 min_y 이상이고 max_y 이하인지 확인
        # 즉, 세로 위치가 지정된 관심 구역 안에 있는지 판단
        y_ok = (y >= self.min_y) and (y <= self.max_y)

        # 두 조건(면적, y범위)을 모두 만족해야 True
        return area_ok and y_ok
    
        # ------------------------------------------
        # 내부 함수
        # ------------------------------------------

    # 캡처 / YOLO 추론 / json에 추가
    def yolo_inference(self):
        """
        YOLO 추론 실행 및 결과를 JSON에 저장
        """
        if self.yolo_model is None:
            print("[YOLO] YOLO 모델이 초기화되지 않았습니다.")
            return
        
        if not hasattr(self, '_last_bgr_img') or self._last_bgr_img is None:
            print("[YOLO] 이미지가 없습니다.")
            return
        
        print("[YOLO] 객체 인식 시작...")
        
        # YOLO 추론 실행
        detections = self.yolo_model.predict(self._last_bgr_img)
        
        if not detections:
            print("[YOLO] 객체를 감지하지 못했습니다.")
            return
        
        print(f"[YOLO] {len(detections)}개 객체 감지")
        
        # 섹터 정보 가져오기
        sector = self.current_sector
        if sector is None and self._last_marker_id is not None:
            sector = self.marker_to_sector.get(self._last_marker_id, "Alpha")
            self.current_sector = sector
        
        if sector is None:
            sector = "Alpha"  # 기본값
        
        # 섹터 변환: sector1-9를 Alpha/Bravo/Charlie로 매핑
        # sector1-3: Alpha, sector4-6: Bravo, sector7-9: Charlie
        dashboard_sector = sector
        if sector.startswith("sector"):
            sector_num = int(sector.replace("sector", ""))
            if 1 <= sector_num <= 5:
                dashboard_sector = "Alpha"
            elif 6 <= sector_num <= 7:
                dashboard_sector = "Bravo"
            elif 8 <= sector_num <= 9:
                dashboard_sector = "Charlie"
        elif sector == "Finish":
            dashboard_sector = "Alpha"  # Finish는 기본적으로 Alpha로 처리
        
        # 객체 타입별 카운트 (신뢰도 필터링)
        detection_counts = {}
        min_confidence = 0.5  # 신뢰도 임계값 (0.5 이상만 인정)
        
        for det in detections:
            class_id = det['class']
            confidence = det['conf']
            
            if class_id < len(YOLO_CLASS_NAMES):
                class_name = YOLO_CLASS_NAMES[class_id].lower()  # 소문자로 변환
            else:
                class_name = f"class_{class_id}"
            
            # 신뢰도가 너무 낮으면 스킵
            if confidence < min_confidence:
                print(f"  - {class_name}: {confidence:.2f} (신뢰도 낮음, 스킵)")
                continue
            
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            print(f"  - {class_name}: {confidence:.2f}")
        
        # 대시보드에 추가
        if self.dashboard:
            for obj_type, count in detection_counts.items():
                self.dashboard.add_detection(dashboard_sector, obj_type, count)
            print(f"[YOLO] {dashboard_sector} 섹터에 감지 결과 추가 완료 (원본 섹터: {sector})")
        
        print("[YOLO] 추론 완료")

    # 화재건물 캡처하기 위한 함수 / 로봇 jetson이 화재건물을 인식하면 해당 아루코태그에 가서 캡처
    def fire_building_capture(self):
        """
        화재 건물 이미지 캡처 및 대시보드 전송
        """
        if not hasattr(self, '_last_bgr_img') or self._last_bgr_img is None:
            print("[FIRE] 이미지가 없습니다.")
            return
        
        # 섹터 정보 가져오기
        sector = self.current_sector
        if sector is None and self._last_marker_id is not None:
            # 마커 ID로 섹터 추정
            marker_sector = self.marker_to_sector.get(self._last_marker_id)
            if marker_sector:
                sector = marker_sector
            else:
                # 화재 섹터 리더에서 확인
                if self.fire_sector_reader:
                    # 마커 ID를 섹터 이름으로 변환 (예: 3 -> "sector3")
                    sector_name = f"sector{self._last_marker_id}"
                    if self.fire_sector_reader.is_fire_building_sector(sector_name):
                        sector = sector_name
        
        if sector is None:
            sector = "unknown"
        
        # 이미지 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = f"fire_building_{sector}_{timestamp}.jpg"
        image_path = self.image_dir / image_filename
        
        cv2.imwrite(str(image_path), self._last_bgr_img)
        print(f"[FIRE] 이미지 저장: {image_path}")
        
        # 대시보드로 이미지 전송
        if self.dashboard:
            self.dashboard.send_dashboard_image(str(image_path))
        
        # 화재 건물 목록 업데이트 (섹터 이름이 "sectorX" 형식인 경우)
        if self.fire_sector_reader and self._last_marker_id is not None:
            sector_name = f"sector{self._last_marker_id}"
            if self.fire_sector_reader.is_fire_building_sector(sector_name):
                # 이미 화재 건물로 등록되어 있음
                print(f"[FIRE] {sector_name}는 이미 화재 건물로 등록되어 있습니다.")
            else:
                # 화재 건물 목록에 추가 (필요한 경우)
                # 주의: 이 부분은 실제 요구사항에 따라 수정 필요
                pass

    # 체크포인트 통과 시 보고하기 위한 함수 / 아루코 태그 : 1, 8, 10
    def checkpoint_passing(self):
        """
        체크포인트 통과 보고
        """
        if self.dashboard is None:
            print("[CHECKPOINT] 대시보드가 초기화되지 않았습니다.")
            return
        
        # 섹터 정보 가져오기
        sector = self.current_sector
        if sector is None and self._last_marker_id is not None:
            sector = self.marker_to_sector.get(self._last_marker_id, "Alpha")
            self.current_sector = sector
        
        if sector is None:
            sector = "Alpha"
        
        # 포인트 추가 (소문자로 변환)
        point_name = sector.lower()
        self.dashboard.add_point(point_name)
        print(f"[CHECKPOINT] {sector} 체크포인트 통과 보고 완료")

    # 포트홀 회피 알고리즘 : 8번 아루코 태그에서 사용예정
    def pothole_avoidance(self):
        
        # 로봇 제어 코드 작성
        # 경기장 특성상 왼쪽으로 회피해야함
        self.tiki.forward(40)
        time.sleep(2.5)
        self.tiki.stop()
    
        # 왼쪽으로 30도 회전부터
        self.tiki.counter_clockwise(40)
        time.sleep(0.5)
        self.tiki.stop()
    
        # 전진(차선 벗어나기)
        self.tiki.forward(40)
        time.sleep(4)
        self.tiki.stop()
    
        # 오른쪽으로 45도 회전
        self.tiki.clockwise(40)
        time.sleep(0.7)
        self.tiki.stop()
    
        # 전진(차선 진입하기)
        self.tiki.forward(40)
        time.sleep(4.5)
        self.tiki.stop()
    
        # 왼쪽으로 30도 회전
        self.tiki.counter_clockwise(40)
        time.sleep(0.5)
        self.tiki.stop()

    # 움직임 / 정해진 시간만큼 직진, 회전 또는 각도 기반 회전
    def _rotate_or_move(self, direction, duration_sec):

        # 직진 출력 : 20 RPM / 회전 출력 : 40 RPM 사용 예정
        if direction == "forward":  
            self.tiki.forward(self.straight_rpm)

        elif direction == "backward":
            self.tiki.backward(self.straight_rpm)

        elif direction == "right":
            self.tiki.clockwise(self.turn_rpm)

        elif direction == "left":
            self.tiki.counter_clockwise(self.turn_rpm)

        else:
            return

        time.sleep(duration_sec)
        self.tiki.stop()
        time.sleep(1)

        # 결과 출력
        # print(f"[ArucoTrigger] Action: {direction} | {value} sec/deg")

    # 아루코 태그가 식별되는지 관찰하고 쿨타임 만족 시 액션수행의 트리거가 되는 함수
    def observe_and_maybe_trigger(self, frame):

        self._last_bgr_img = frame

        if self.mode != "LANE_FOLLOW":
            return

        now = time.time()
        dets = self.detector.detect_ids(frame)

        if not dets:
            self._consec = {}
            self._last_marker_id = None
            return
        
        dets = [d for d in dets if self._gate(d)]
        if not dets:
            self._consec = {}
            self._last_marker_id = None
            return

        det = max(dets, key=lambda x: x["area"])
        mid = det["id"]
        self._last_marker_id = mid

        # 섹터 정보 업데이트
        if mid in self.marker_to_sector:
            self.current_sector = self.marker_to_sector[mid]
        
        # 화재 섹터 정보 업데이트 (주기적으로 확인만, rules는 수정하지 않음)
        if self.fire_sector_reader:
            self.fire_sector_reader.update_fire_sectors()

        # 연속 검출
        self._consec[mid] = self._consec.get(mid, 0) + 1
        for k in list(self._consec.keys()):
            if k != mid:
                self._consec[k] = 0

        if self._consec[mid] < self.required_consecutive:
            return

        # 쿨다운 체크
        last = self.last_trigger_times.get(mid, 0)
        cooldown = self.cooldown_per_id.get(mid, self.cooldown_default)
        if now - last < cooldown:
            return

        # 등장 횟수
        nth = self.seen_counts.get(mid, 0) + 1
        self.seen_counts[mid] = nth

        print(f"[ArucoTrigger] Tag {mid} detected (nth={nth}, sector={self.current_sector})")

        # 아루코 태그 인식 후 액션수행을 위해 모드를 변경합니다.
        if mid in self.rules and nth in self.rules[mid]:
            self.pending_actions = list(self.rules[mid][nth])
            self.mode = "EXECUTE_ACTION"
            self.last_trigger_times[mid] = now
            self._consec = {}

    # ------------------------------------------
    # 매 프레임 호출
    # ------------------------------------------

    def step(self):

        # 차선주행일 때는 종료
        if self.mode != "EXECUTE_ACTION":
            return

        # pending_actions 실행 전에 화재 건물인지 확인하고 FIRE 액션 동적 추가
        if self._last_marker_id is not None and self.fire_sector_reader:
            # 최신 화재 섹터 정보 업데이트
            self.fire_sector_reader.update_fire_sectors()
            
            # 마커 ID -> 섹터 이름 변환
            # 마커 2 -> sector1, 마커 3 -> sector2, 마커 4 -> sector3, ...
            if 2 <= self._last_marker_id <= 12:
                # sector 마커들만 확인 (2-12번)
                sector_num = self._last_marker_id - 1  # 마커 2 -> sector1, 마커 3 -> sector2
                sector_name = f"sector{sector_num}"
                
                if self.fire_sector_reader.is_fire_building_sector(sector_name):
                    # 화재 건물이면 FIRE 액션을 맨 앞에 추가 (아직 실행 안 했으면)
                    if not any(action[0] == "FIRE" for action in self.pending_actions):
                        self.pending_actions.insert(0, ("FIRE", 0.0))
                        print(f"[FIRE] {sector_name} 화재 건물 감지 - FIRE 액션 자동 추가")

        # pending_actions가 있을 동안 계속 실행
        while self.pending_actions:

            action, val = self.pending_actions.pop(0)

            self.tiki.stop()

            # 화재건물 캡처
            if action == "FIRE":
                self.fire_building_capture()

            # YOLO 추론
            elif action == "YOLO":
                self.yolo_inference()

            # 포트홀 회피
            elif action == "POTHOLE_AVOIDANCE":
                self.pothole_avoidance()

            # 체크포인트 통과
            elif action == "CHECKPOINT_PASSING":
                self.checkpoint_passing()

            # 기본 동작 (직진·후진·좌회전·우회전 등)
            else:
                self._rotate_or_move(action, val)

        # 모든 액션 끝났으면 차선주행 모드 복귀
        self.mode = "LANE_FOLLOW"