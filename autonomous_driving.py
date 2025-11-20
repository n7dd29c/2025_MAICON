"""
자율주행 메인 실행 파일
모든 프레임에서 모든 인식 실행:
- 모든 프레임: 차선 인식 및 자율주행 제어 (실시간)
- 모든 프레임: YOLO 객체 인식
- 모든 프레임: ArUco 마커 인식
- 모든 프레임: QR 코드 인식
"""
import cv2
import time
import argparse
import numpy as np
from typing import Optional
from lane_detector import LaneDetector
from autonomous_control import AutonomousController
from object_detector import ObjectDetector
from avoidance_planner import AvoidancePlanner
from aruco_detector import ArUcoDetector
from qr_detector import QRDetector
from dashboard_comm import DashboardComm
from config import USE_BIRD_VIEW, USE_GSTREAMER, USE_ARUCO, USE_QR, USE_DASHBOARD, USE_DASHBOARD


class AutonomousDriving:
    """자율주행 메인 클래스"""
    
    def __init__(self,
                 camera_id: int = 0,
                 image_width: int = 640,
                 image_height: int = 480,
                 fps_target: int = 15):
        """
        Args:
            camera_id: 카메라 ID
            image_width: 이미지 너비
            image_height: 이미지 높이
            fps_target: 목표 FPS
        """
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.fps_target = fps_target
        
        # 차선 인식기 초기화
        self.lane_detector = LaneDetector(
            image_width=image_width,
            image_height=image_height,
            use_bird_view=USE_BIRD_VIEW
        )
        
        # 제어기 초기화
        from config import USE_TIKI, MOTOR_MODE
        self.controller = AutonomousController(
            max_steering_angle=30.0,
            kp=0.5,
            kd=0.1,
            use_tiki=USE_TIKI,
            motor_mode=MOTOR_MODE
        )
        
        # 객체 디텍터 초기화 (포트홀 감지)
        self.object_detector = ObjectDetector(
            image_width=image_width,
            image_height=image_height,
            use_bird_view=USE_BIRD_VIEW
        )
        
        # 회피 계획기 초기화
        self.avoidance_planner = AvoidancePlanner(
            image_width=image_width,
            image_height=image_height
        )
        
        # ArUco 마커 감지기 초기화 (베이스라인 코드 기준)
        if USE_ARUCO:
            from config import ARUCO_DICTIONARY
            self.aruco_detector = ArUcoDetector(dictionary_id=ARUCO_DICTIONARY)
        else:
            self.aruco_detector = None
        
        # QR 코드 감지기 초기화
        if USE_QR:
            self.qr_detector = QRDetector()
        else:
            self.qr_detector = None
        
        # 대시보드 통신 초기화
        if USE_DASHBOARD:
            from config import DASHBOARD_SERVER_URL, MISSION_CODE
            self.dashboard = DashboardComm(
                server_url=DASHBOARD_SERVER_URL,
                mission_code=MISSION_CODE
            )
            # 초기 데이터 전송 (대시보드 초기화)
            self.dashboard.send_dashboard_json()
            print("대시보드 통신 초기화 완료")
        else:
            self.dashboard = None
        
        # 카메라 초기화
        self.cap: Optional[cv2.VideoCapture] = None
        
        # 프레임 카운터
        self.frame_count = 0
        
        # 성능 측정
        self.fps_history = []
        
        # 차선 정보 저장 (회피 계획에 사용)
        self.last_left_lane = None
        self.last_right_lane = None
        self.last_offset = 0.0
        
        # 차선 미감지 카운터 (연속 미감지 추적)
        self.lane_lost_count = 0
        self.max_lane_lost_frames = 30  # 연속 30프레임 미감지 시 정지
        
        # 회피 명령 저장
        self.current_avoidance_command = None
        
        # ArUco 명령 저장
        self.current_aruco_command = None
        self.aruco_command_start_time = None
        self.aruco_command_duration = 0.0
        
        # QR 코드 감지 상태
        self.last_qr_data = None
        self.qr_led_start_time = None
        
        # 대시보드 통신 상태
        self.last_dashboard_update = 0.0
        self.detection_counts = {}  # 객체 타입별 카운트 저장
        
        # 현재 로봇 섹터 (ArUco 마커 감지 시 자동 업데이트)
        from config import DASHBOARD_SECTOR
        self.current_sector = DASHBOARD_SECTOR
        
    def initialize_camera(self, use_gstreamer: bool = True) -> bool:
        """
        카메라 초기화
        Jetson Nano CSI 카메라의 경우 GStreamer 파이프라인 사용
        """
        if use_gstreamer:
            # Jetson Nano CSI 카메라용 GStreamer 파이프라인
            # 베이스라인 코드와 동일한 파이프라인 사용
            pipeline = (
                f"nvarguscamerasrc ! video/x-raw(memory:NVMM), "
                f"width={self.image_width}, height={self.image_height}, "
                f"format=NV12, framerate={self.fps_target}/1 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            
            # 최대 3번 재시도
            max_retries = 3
            for attempt in range(max_retries):
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                # 카메라가 열릴 때까지 대기
                time.sleep(1.0)
                
                if self.cap.isOpened():
                    # 프레임 읽기 테스트
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"카메라 초기화 완료: {self.image_width}x{self.image_height} (GStreamer: True)")
                        return True
                    else:
                        print(f"경고: 프레임 읽기 실패 (시도 {attempt + 1}/{max_retries})")
                        self.cap.release()
                        time.sleep(0.5)
                else:
                    print(f"경고: 카메라 열기 실패 (시도 {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
            
            print("카메라 초기화 실패: 여러 번 시도했지만 실패했습니다.")
            print("해결 방법:")
            print("1. 다른 프로세스에서 카메라를 사용 중인지 확인: ps aux | grep python")
            print("2. Jetson Nano를 재부팅")
            print("3. config.py에서 USE_GSTREAMER = False로 설정하여 일반 카메라 모드 시도")
            return False
        else:
            # 일반 USB 카메라
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if self.cap.isOpened():
                # 카메라 설정
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
                
                # 프레임 읽기 테스트
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"카메라 초기화 완료: {self.image_width}x{self.image_height} (GStreamer: False)")
                    return True
                else:
                    print("경고: 카메라 초기화는 되었지만 프레임 읽기 실패")
                    return False
            else:
                print(f"카메라를 열 수 없습니다. (카메라 ID: {self.camera_id})")
                return False
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        프레임 처리 (모든 프레임에서 모든 인식 실행)
        - 모든 프레임: 차선 인식, YOLO 객체 인식, ArUco 마커, QR 코드 인식
        - 실시간 제어를 위해 모든 인식을 매 프레임마다 실행
        
        Returns:
            (processed_frame, control_command): 처리된 프레임과 제어 명령
        """
        self.frame_count += 1
        vis_frame = frame.copy()
        status_messages = []
        
        # 모든 프레임: YOLO 객체 인식
        from config import USE_YOLO
        if USE_YOLO:
            # YOLO 사용 (ONNX) - 모든 객체 감지
            avoidance_objects, all_objects = self.object_detector.detect_objects(frame)
            potholes = avoidance_objects  # 회피 대상 객체
            
            # 회피 계획 생성 (포트홀 회피 활성화 시에만)
            from config import USE_POTHOLE_AVOIDANCE
            if USE_POTHOLE_AVOIDANCE and potholes:
                # 가장 가까운 포트홀 선택 (이미지 하단에 가까운 것, y 좌표가 큰 것)
                closest_pothole = max(potholes, key=lambda p: p['center'][1])
                
                avoidance_command = self.avoidance_planner.plan_avoidance(
                    pothole=closest_pothole,
                    left_lane=self.last_left_lane,
                    right_lane=self.last_right_lane,
                    current_offset=self.last_offset
                )
                self.current_avoidance_command = avoidance_command
            else:
                # 정상 주행 (회피 비활성화 또는 포트홀 없음)
                self.current_avoidance_command = {'action': 'normal'}
            
            # 모든 객체 시각화
            if all_objects:
                vis_frame = self.object_detector.draw_objects(vis_frame, all_objects, show_class_name=True)
                avoidance_count = len(potholes)
                total_count = len(all_objects)
                status_messages.append(f"YOLO: {total_count}개 (회피: {avoidance_count}개)")
                
                # 대시보드로 객체 감지 정보 전송
                if USE_DASHBOARD and self.dashboard is not None:
                    from config import DASHBOARD_UPDATE_INTERVAL
                    current_time = time.time()
                    # 일정 간격마다 업데이트
                    if current_time - self.last_dashboard_update >= DASHBOARD_UPDATE_INTERVAL:
                        # 객체 타입별 카운트 집계
                        detection_counts = {}
                        for obj in all_objects:
                            obj_type = obj.get('class_name', 'unknown').lower()
                            detection_counts[obj_type] = detection_counts.get(obj_type, 0) + 1
                        
                        # 대시보드 형식으로 변환
                        detections = [
                            {"type": obj_type, "count": count}
                            for obj_type, count in detection_counts.items()
                        ]
                        
                        # 대시보드로 전송 (현재 로봇 섹터 사용)
                        self.dashboard.update_detection(self.current_sector, detections)
                        self.last_dashboard_update = current_time
                        print(f"대시보드 업데이트: {self.current_sector} 섹터, {len(detections)}개 객체 타입")
            else:
                status_messages.append("YOLO: 객체 없음")
        else:
            # 전통적인 CV 방법 (포트홀만)
            potholes = self.object_detector.detect_potholes(frame)
            if potholes:
                vis_frame = self.object_detector.draw_potholes(vis_frame, potholes)
                status_messages.append(f"포트홀: {len(potholes)}개")
                
                # 회피 계획 생성 (포트홀 회피 활성화 시에만)
                from config import USE_POTHOLE_AVOIDANCE
                if USE_POTHOLE_AVOIDANCE:
                    closest_pothole = max(potholes, key=lambda p: p['center'][1])
                    avoidance_command = self.avoidance_planner.plan_avoidance(
                        pothole=closest_pothole,
                        left_lane=self.last_left_lane,
                        right_lane=self.last_right_lane,
                        current_offset=self.last_offset
                    )
                    self.current_avoidance_command = avoidance_command
                else:
                    # 회피 비활성화 - 정상 주행
                    self.current_avoidance_command = {'action': 'normal'}
            else:
                self.current_avoidance_command = {'action': 'normal'}
                status_messages.append("포트홀: 없음")
        
        # 모든 프레임: ArUco 마커 감지
        aruco_markers = []
        if USE_ARUCO and self.aruco_detector is not None:
            aruco_command, aruco_markers = self.aruco_detector.detect_markers(frame)
            
            if aruco_command is not None:
                marker_id = aruco_command.get('marker_id')
                action = aruco_command.get('action')
                
                # 대시보드로 ArUco 마커 감지 정보 전송 (통신 보고)
                if USE_DASHBOARD and self.dashboard is not None:
                    # 마커 ID별 섹터 설정 확인
                    from config import ARUCO_MARKER_SECTORS, DASHBOARD_SECTOR
                    marker_sector = ARUCO_MARKER_SECTORS.get(marker_id, None)
                    
                    # 마커에 섹터가 설정되어 있으면 로봇 섹터 업데이트
                    if marker_sector is not None:
                        old_sector = self.current_sector
                        self.current_sector = marker_sector
                        print(f"로봇 섹터 변경: {old_sector} → {marker_sector} (ArUco ID={marker_id})")
                    else:
                        # 마커에 섹터가 없으면 현재 섹터 유지
                        marker_sector = self.current_sector
                    
                    # 포인트 추가 (ArUco 마커 감지 시)
                    point_name = f"aruco_{marker_id}_{action}"
                    self.dashboard.add_point(point_name)
                    print(f"대시보드 전송: ArUco ID={marker_id}, Action={action}, Sector={marker_sector}")
                    
                    # Finish 마커(ID 13) 감지 시 즉시 대시보드로 전송
                    if marker_sector == "Finish" or marker_id == 13:
                        print(f"임무 종료 마커 감지 (ID={marker_id}) - 대시보드로 즉시 전송")
                        self.dashboard.send_dashboard_json()
                
                # 주행 제어 여부 확인
                from config import ARUCO_CONTROL_ENABLED
                if ARUCO_CONTROL_ENABLED:
                    # 주행 제어 활성화 시에만 명령 저장
                    self.current_aruco_command = aruco_command
                    from config import ARUCO_TURN_DURATION
                    self.aruco_command_start_time = time.time()
                    self.aruco_command_duration = ARUCO_TURN_DURATION
                    print(f"ArUco 주행 명령 저장: {action}, ID={marker_id}")
                else:
                    # 통신 보고만 수행 (주행 제어 안 함)
                    self.current_aruco_command = None
                    print(f"ArUco 마커 감지 (통신 보고만): ID={marker_id}, Action={action}")
            else:
                # ArUco 명령이 없으면 이전 명령 지속 시간 확인 (주행 제어 활성화 시에만)
                from config import ARUCO_CONTROL_ENABLED
                if ARUCO_CONTROL_ENABLED and self.aruco_command_start_time is not None:
                    elapsed = time.time() - self.aruco_command_start_time
                    if elapsed >= self.aruco_command_duration:
                        # 지속 시간 경과 시 명령 해제
                        self.current_aruco_command = None
                        self.aruco_command_start_time = None
            
            # ArUco 마커 시각화 (감지된 마커가 있으면)
            if aruco_markers:
                vis_frame = self.aruco_detector.draw_markers(vis_frame, aruco_markers)
                status_messages.append(f"ArUco: {len(aruco_markers)}개")
        
        # 모든 프레임: QR 코드 감지 (LED 제어를 위해)
        qr_codes = []
        if USE_QR and self.qr_detector is not None:
            qr_codes = self.qr_detector.detect(frame)
            
            if qr_codes:
                # QR 코드 시각화
                vis_frame = self.qr_detector.draw_codes(vis_frame, qr_codes)
                status_messages.append(f"QR: {len(qr_codes)}개")
                
                # QR 코드 감지 시 LED 제어
                for qr in qr_codes:
                    qr_data = qr['data']
                    # 새로운 QR 코드 감지 시 LED 켜기
                    if qr_data != self.last_qr_data:
                        self.last_qr_data = qr_data
                        # LED 켜기 (QR 코드 데이터에서 색상 파싱)
                        try:
                            # QR 코드 데이터에서 색상 파싱 (ID_SHAPE_COLOR 형식)
                            if qr_data.startswith("ID_"):
                                parts = qr_data.split("_")
                                if len(parts) >= 3:
                                    color_str = parts[2].upper()  # R, G, B
                                    # LED 색상 매핑
                                    if color_str == 'R':
                                        led_color = "red"
                                    elif color_str == 'G':
                                        led_color = "green"
                                    elif color_str == 'B':
                                        led_color = "blue"
                                    else:
                                        led_color = "green"  # 기본값
                                else:
                                    led_color = "green"  # 기본값
                            else:
                                led_color = "green"  # 기본값
                            
                            self.controller.set_led(led_color)
                            print(f"QR 코드 감지: {qr_data} - LED 켜짐 (색상: {led_color})")
                        except Exception as e:
                            print(f"QR LED 제어 오류: {e}")
                        
                        from config import QR_LED_DURATION
                        self.qr_led_start_time = time.time()
                        
                        # 대시보드로 QR 코드 감지 정보 전송
                        if USE_DASHBOARD and self.dashboard is not None:
                            # 포인트 추가 (QR 코드 감지 시)
                            point_name = f"qr_{qr_data[:10]}"  # QR 데이터의 처음 10자만 사용
                            self.dashboard.add_point(point_name)
                            print(f"대시보드 포인트 추가: {point_name} (QR: {qr_data})")
            else:
                # QR 코드 미감지 시 LED 끄기 (지속 시간 경과 후)
                if self.qr_led_start_time is not None:
                    from config import QR_LED_DURATION
                    elapsed = time.time() - self.qr_led_start_time
                    if elapsed >= QR_LED_DURATION:
                        try:
                            self.controller.set_led_off()
                            print("QR LED 끄기 (지속 시간 경과)")
                        except Exception as e:
                            print(f"QR LED 끄기 오류: {e}")
                        self.qr_led_start_time = None
                        self.last_qr_data = None
        
        # 상태 메시지 표시
        if status_messages:
            status_text = " | ".join(status_messages)
            cv2.putText(vis_frame, status_text,
                       (10, self.image_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 모든 프레임: 차선 인식 및 자율주행 제어 (실시간 제어를 위해)
        # 차선 감지 (중앙선 포함)
        left_lane, right_lane, center_lane = self.lane_detector.detect_lanes(frame)
        
        # 차선 미감지 처리
        if center_lane is None and (left_lane is None or right_lane is None):
            self.lane_lost_count += 1
            
            # 초기 몇 프레임은 대기
            if self.frame_count <= 10:
                print(f"차선 감지 대기 중... (프레임 {self.frame_count})")
                return vis_frame, None
            
            # 이전 차선 정보가 있으면 사용하여 계속 주행
            if self.last_left_lane is not None and self.last_right_lane is not None:
                # 이전 차선 정보 사용
                left_lane = self.last_left_lane
                right_lane = self.last_right_lane
                center_lane = None
                
                # 연속 미감지가 임계값을 넘으면 정지
                if self.lane_lost_count >= self.max_lane_lost_frames:
                    print(f"경고: 차선 연속 미감지 {self.lane_lost_count}프레임 - 정지")
                    stop_command = {
                        'steering_angle': 0.0,
                        'speed': 0.0,
                        'is_safe': False,
                        'safety_message': f'차선 연속 미감지 - 정지 ({self.lane_lost_count}프레임)',
                        'offset': self.last_offset,
                        'avoidance_mode': False,
                        'aruco_mode': False
                    }
                    vis_frame = frame.copy()
                    cv2.putText(vis_frame, f"차선 미감지 - 정지 ({self.lane_lost_count})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    return vis_frame, stop_command
                else:
                    # 이전 차선 정보로 계속 주행 (느린 속도)
                    if self.frame_count % 10 == 0:  # 10프레임마다 한 번만 출력
                        print(f"경고: 차선 미감지 - 이전 차선 정보 사용 (연속 {self.lane_lost_count}프레임)")
            else:
                # 이전 차선 정보도 없으면 정지
                if self.lane_lost_count >= self.max_lane_lost_frames:
                    print(f"경고: 차선 미감지 - 정지 (프레임 {self.frame_count})")
                    stop_command = {
                        'steering_angle': 0.0,
                        'speed': 0.0,
                        'is_safe': False,
                        'safety_message': '차선 미감지 - 정지',
                        'offset': 0.0,
                        'avoidance_mode': False,
                        'aruco_mode': False
                    }
                    vis_frame = frame.copy()
                    cv2.putText(vis_frame, "차선 미감지 - 정지",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    return vis_frame, stop_command
        else:
            # 차선 감지 성공: 카운터 리셋
            self.lane_lost_count = 0
        
        # 차선 정보 저장 (회피 계획에 사용)
        self.last_left_lane = left_lane
        self.last_right_lane = right_lane
        
        # 중심점 계산 (중앙선 우선)
        center_point = self.lane_detector.calculate_center(left_lane, right_lane, center_lane)
        
        # 오프셋 계산
        offset = self.lane_detector.calculate_offset(center_point)
        self.last_offset = offset
        
        # 곡선 정보 가져오기
        is_curve, curve_radius, curve_direction = self.lane_detector.get_curve_info()
        
        # 제어 명령 생성 (우선순위: 포트홀 회피 > ArUco 명령 > 차선 추종)
        # ArUco 주행 제어가 비활성화된 경우 None 전달
        from config import ARUCO_CONTROL_ENABLED
        aruco_command_for_control = self.current_aruco_command if ARUCO_CONTROL_ENABLED else None
        
        control_command = self.controller.get_control_command(
            offset=offset,
            left_lane=left_lane,
            right_lane=right_lane,
            center_lane=center_lane,  # 중앙선 정보 전달 (더 강한 중앙 정렬)
            image_width=self.image_width,
            is_curve=is_curve,  # 곡선 구간 여부
            curve_radius=curve_radius,  # 곡선 반경
            curve_direction=curve_direction,  # 곡선 방향 (-1: 왼쪽, +1: 오른쪽)
            avoidance_command=self.current_avoidance_command,
            aruco_command=aruco_command_for_control
        )
        
        # 차선 미감지 시 속도 감소 (이전 차선 정보 사용 중인 경우)
        if self.lane_lost_count > 0 and self.lane_lost_count < self.max_lane_lost_frames:
            # 차선 미감지 시 속도를 70%로 감소
            control_command['speed'] = control_command['speed'] * 0.7
            control_command['safety_message'] = f"차선 미감지 - 이전 정보 사용 (속도 감소, {self.lane_lost_count}프레임)"
        
        # 시각화 (중앙선 포함)
        vis_frame = self.lane_detector.draw_lanes(
            vis_frame, left_lane, right_lane, center_point, center_lane
        )
        
        # 제어 정보 오버레이
        vis_frame = self._draw_control_info(vis_frame, control_command)
        
        return vis_frame, control_command
    
    def _draw_control_info(self, frame: np.ndarray, control: dict) -> np.ndarray:
        """제어 정보 시각화"""
        if control is None:
            return frame
        
        # 정보 텍스트
        info_text = [
            f"Steering: {control['steering_angle']:.1f} deg",
            f"Speed: {control['speed']:.2f}",
            f"Offset: {control['offset']:.1f} px",
            f"Status: {control['safety_message']}"
        ]
        
        # 안전 상태에 따른 색상
        color = (0, 255, 0) if control['is_safe'] else (0, 0, 255)
        
        # 텍스트 그리기
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 프레임 번호
        cv2.putText(frame, f"Frame: {self.frame_count} (모든 프레임 처리)",
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def execute_control(self, control_command: dict):
        """제어 명령 실행 (실제 하드웨어 제어)"""
        if control_command is None:
            return
        
        steering_angle = control_command['steering_angle']
        speed = control_command['speed']
        safety_message = control_command.get('safety_message', '')
        
        # 차선 미감지 시 완전 정지
        if '차선 미감지' in safety_message:
            print(f"정지: {safety_message}")
            self.controller.stop_motors()
            return
        
        # 안전하지 않으면 경고 출력
        if not control_command['is_safe']:
            print(f"경고: {safety_message}")
            # 차선 이탈 위험일 때는 조향을 실행하여 복귀하도록 함
            # 속도만 줄임
            speed = min(speed, self.controller.min_speed * 0.5)  # 최소 속도의 50%로 감속
        
        # TikiMini API로 모터 제어 (조향은 항상 실행)
        self.controller.execute_motor_control(steering_angle, speed)
    
    def run(self, show_preview: bool = True, save_video: bool = False):
        """메인 실행 루프"""
        if not self.initialize_camera(use_gstreamer=USE_GSTREAMER):
            return
        
        # 비디오 저장 설정
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                'autonomous_driving_output.avi',
                fourcc, self.fps_target,
                (self.image_width, self.image_height)
            )
        
        print("자율주행 시작 (모든 프레임에서 차선/객체/QR/ArUco 인식)")
        print("종료: 'q' 키 누르기")
        
        try:
            while True:
                start_time = time.time()
                
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임 읽기 실패")
                    break
                
                # 프레임 180도 회전 (뒤집힌 영상 보정)
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # 프레임 처리
                processed_frame, control_command = self.process_frame(frame)
                
                # 제어 명령 실행
                if control_command:
                    self.execute_control(control_command)
                
                # FPS 계산
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # FPS 표시
                cv2.putText(processed_frame, f"FPS: {avg_fps:.1f}",
                           (self.image_width - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 비디오 저장
                if save_video and video_writer:
                    video_writer.write(processed_frame)
                
                # 화면 표시
                if show_preview:
                    cv2.imshow('Autonomous Driving', processed_frame)
                    
                    # 'q' 키로 종료
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 목표 FPS 맞추기
                elapsed = time.time() - start_time
                target_time = 1.0 / self.fps_target
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
        
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
        
        finally:
            # 모터 정지
            self.controller.stop_motors()
            
            # 정리
            if self.cap:
                self.cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            if self.fps_history:
                print(f"평균 FPS: {sum(self.fps_history) / len(self.fps_history):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Jetson Nano 자율주행 시스템')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (기본값: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='이미지 너비 (기본값: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='이미지 높이 (기본값: 480)')
    parser.add_argument('--fps', type=int, default=15,
                       help='목표 FPS (기본값: 15)')
    parser.add_argument('--no-preview', action='store_true',
                       help='화면 미리보기 비활성화')
    parser.add_argument('--save-video', action='store_true',
                       help='비디오 저장')
    
    args = parser.parse_args()
    
    # 자율주행 시스템 초기화 및 실행
    driving = AutonomousDriving(
        camera_id=args.camera,
        image_width=args.width,
        image_height=args.height,
        fps_target=args.fps
    )
    
    driving.run(
        show_preview=not args.no_preview,
        save_video=args.save_video
    )


if __name__ == '__main__':
    main()

