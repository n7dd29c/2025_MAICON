"""
자율주행 메인 실행 파일
홀수 프레임에서 차선 인식 및 자율주행 제어 수행
짝수 프레임에서 포트홀 감지 및 회피 계획
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
from config import USE_BIRD_VIEW, USE_GSTREAMER, USE_ARUCO


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
        
        # ArUco 마커 감지기 초기화
        if USE_ARUCO:
            from config import ARUCO_DICTIONARY
            self.aruco_detector = ArUcoDetector(dictionary_id=ARUCO_DICTIONARY)
        else:
            self.aruco_detector = None
        
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
        
        # 회피 명령 저장
        self.current_avoidance_command = None
        
        # ArUco 명령 저장
        self.current_aruco_command = None
        self.aruco_command_start_time = None
        self.aruco_command_duration = 0.0
        
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
        프레임 처리
        - 홀수 프레임: 차선 인식 및 자율주행 제어
        - 짝수 프레임: 포트홀 감지 및 회피 계획
        
        Returns:
            (processed_frame, control_command): 처리된 프레임과 제어 명령
        """
        self.frame_count += 1
        
        # 짝수 프레임: 포트홀 감지 및 ArUco 마커 감지
        if self.frame_count % 2 == 0:
            # 포트홀 감지 임시 비활성화 (오탐 방지 - 차선 추종 테스트용)
            potholes = []
            # potholes = self.object_detector.detect_potholes(frame)  # 주석 처리
            
            # 정상 주행만 (포트홀 회피 비활성화)
            self.current_avoidance_command = {'action': 'normal'}
            
            # ArUco 마커 감지
            aruco_markers = []
            if USE_ARUCO and self.aruco_detector is not None:
                aruco_command, aruco_markers = self.aruco_detector.detect_markers(frame)
                
                if aruco_command is not None:
                    # ArUco 명령이 감지되면 저장
                    self.current_aruco_command = aruco_command
                    from config import ARUCO_TURN_DURATION
                    self.aruco_command_start_time = time.time()
                    self.aruco_command_duration = ARUCO_TURN_DURATION
                else:
                    # ArUco 명령이 없으면 이전 명령 지속 시간 확인
                    if self.aruco_command_start_time is not None:
                        elapsed = time.time() - self.aruco_command_start_time
                        if elapsed >= self.aruco_command_duration:
                            # 지속 시간 경과 시 명령 해제
                            self.current_aruco_command = None
                            self.aruco_command_start_time = None
            
            # 시각화 (차선 및 ArUco 마커 표시)
            vis_frame = frame.copy()
            
            # 차선 정보 표시
            if self.last_left_lane is not None and self.last_right_lane is not None:
                # 중앙선도 포함하여 계산
                center_point = self.lane_detector.calculate_center(
                    self.last_left_lane, self.last_right_lane, None
                )
                vis_frame = self.lane_detector.draw_lanes(
                    vis_frame, self.last_left_lane, self.last_right_lane, center_point, None
                )
            
            # ArUco 마커 시각화
            if USE_ARUCO and self.aruco_detector is not None and aruco_markers:
                vis_frame = self.aruco_detector.draw_markers(vis_frame, aruco_markers)
            
            # 상태 메시지 표시
            status_msg = "포트홀 감지 비활성화 (테스트 모드)"
            if self.current_aruco_command:
                status_msg += f" | ArUco: {self.current_aruco_command.get('action', 'unknown')}"
            cv2.putText(vis_frame, status_msg,
                       (10, self.image_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return vis_frame, None
        
        # 홀수 프레임: 차선 인식 및 자율주행 제어
        # 차선 감지 (중앙선 포함)
        left_lane, right_lane, center_lane = self.lane_detector.detect_lanes(frame)
        
        # 초기 차선 감지 실패 시 대기 (처음 몇 프레임)
        # 중앙선이 없고 양쪽 차선도 없으면 대기
        if (center_lane is None and (left_lane is None or right_lane is None)) and self.frame_count <= 10:
            # 초기 상태에서는 차선 감지 대기
            print(f"차선 감지 대기 중... (프레임 {self.frame_count})")
            return frame, None
        
        # 차선 정보 저장 (회피 계획에 사용)
        self.last_left_lane = left_lane
        self.last_right_lane = right_lane
        
        # 중심점 계산 (중앙선 우선)
        center_point = self.lane_detector.calculate_center(left_lane, right_lane, center_lane)
        
        # 오프셋 계산
        offset = self.lane_detector.calculate_offset(center_point)
        self.last_offset = offset
        
        # 곡선 정보 가져오기
        is_curve, curve_radius = self.lane_detector.get_curve_info()
        
        # 제어 명령 생성 (우선순위: 포트홀 회피 > ArUco 명령 > 차선 추종)
        control_command = self.controller.get_control_command(
            offset=offset,
            left_lane=left_lane,
            right_lane=right_lane,
            center_lane=center_lane,  # 중앙선 정보 전달 (더 강한 중앙 정렬)
            image_width=self.image_width,
            is_curve=is_curve,  # 곡선 구간 여부
            curve_radius=curve_radius,  # 곡선 반경
            avoidance_command=self.current_avoidance_command,
            aruco_command=self.current_aruco_command
        )
        
        # 시각화 (중앙선 포함)
        vis_frame = self.lane_detector.draw_lanes(
            frame, left_lane, right_lane, center_point, center_lane
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
        cv2.putText(frame, f"Frame: {self.frame_count} (홀수만 처리)",
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def execute_control(self, control_command: dict):
        """제어 명령 실행 (실제 하드웨어 제어)"""
        if control_command is None:
            return
        
        steering_angle = control_command['steering_angle']
        speed = control_command['speed']
        
        # 안전하지 않으면 경고 출력하고 속도만 줄임 (조향은 계속 실행)
        if not control_command['is_safe']:
            print(f"경고: {control_command['safety_message']}")
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
        
        print("자율주행 시작 (홀수 프레임만 처리)")
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

