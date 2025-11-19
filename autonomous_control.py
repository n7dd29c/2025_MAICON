"""
자율주행 제어 모듈
차선 인식 결과를 바탕으로 조향 및 속도 제어
포트홀 회피 기능 포함
TikiMini 로봇 제어 통합
"""
import numpy as np
from typing import Optional, Tuple, Dict
from lane_detector import LaneDetector

# TikiMini API (선택적 import)
try:
    from tiki.mini import TikiMini
    TIKI_AVAILABLE = True
except ImportError:
    TIKI_AVAILABLE = False
    print("경고: TikiMini API를 사용할 수 없습니다. 제어 명령만 생성합니다.")


class AutonomousController:
    """자율주행 제어 클래스"""
    
    def __init__(self,
                 max_steering_angle: float = 30.0,
                 kp: float = 0.5,
                 kd: float = 0.1,
                 max_speed: float = 0.5,
                 min_speed: float = 0.2,
                 use_tiki: bool = True,
                 motor_mode: str = "PID"):
        """
        Args:
            max_steering_angle: 최대 조향 각도 (도)
            kp: 비례 제어 게인
            kd: 미분 제어 게인
            max_speed: 최대 속도 (0.0 ~ 1.0)
            min_speed: 최소 속도 (0.0 ~ 1.0)
            use_tiki: TikiMini API 사용 여부
            motor_mode: 모터 모드 ("PWM" 또는 "PID")
        """
        self.max_steering_angle = max_steering_angle
        self.kp = kp
        self.kd = kd
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.use_tiki = use_tiki and TIKI_AVAILABLE
        
        # TikiMini 초기화
        if self.use_tiki:
            try:
                self.tiki = TikiMini()
                if motor_mode == "PID":
                    self.tiki.set_motor_mode(self.tiki.MOTOR_MODE_PID)
                else:
                    self.tiki.set_motor_mode(self.tiki.MOTOR_MODE_PWM)
                print("TikiMini 초기화 완료")
            except Exception as e:
                print(f"TikiMini 초기화 실패: {e}")
                self.use_tiki = False
                self.tiki = None
        else:
            self.tiki = None
        
        # 이전 오프셋 (미분 제어용)
        self.prev_offset = 0.0
        self.prev_steering_angle = 0.0  # 조향 각도 스무딩용
        
        # 안전 임계값
        self.safe_offset_threshold = 50  # 픽셀 단위
        
        # 회피 명령 (외부에서 설정)
        self.avoidance_command: Optional[Dict] = None
        
        # 모터 제어 파라미터
        self.base_speed_range = 127  # TikiMini 모터 속도 범위 (-127 ~ 127)
        from config import TURN_SENSITIVITY, STEERING_DEADZONE
        self.turn_sensitivity = TURN_SENSITIVITY  # 조향 민감도
        self.deadzone = STEERING_DEADZONE  # 데드존
        
    def calculate_steering(self, offset: float, image_width: int = 640, has_center_lane: bool = False, 
                          is_curve: bool = False, curve_radius: Optional[float] = None) -> float:
        """
        조향 각도 계산 (PID 제어 + 스무딩 + 곡선 보정)
        
        Args:
            offset: 차선 중심으로부터의 오프셋 (픽셀)
                   양수: 차선 중심이 오른쪽에 있음 -> 왼쪽으로 조향 필요
                   음수: 차선 중심이 왼쪽에 있음 -> 오른쪽으로 조향 필요
            image_width: 이미지 너비
            has_center_lane: 중앙선 감지 여부 (True면 더 강하게 반응)
            is_curve: 곡선 구간 여부
            curve_radius: 곡선 반경 (픽셀, 작을수록 급커브)
            
        Returns:
            steering_angle: 조향 각도 (-max_steering_angle ~ max_steering_angle)
                          양수: 오른쪽 회전, 음수: 왼쪽 회전
        """
        # 중앙선이 있으면 데드존을 더 작게 (더 민감하게 반응)
        deadzone = self.deadzone // 2 if has_center_lane else self.deadzone
        
        # 데드존 적용 (작은 오프셋 무시)
        if abs(offset) < deadzone:
            offset = 0.0
        
        # 정규화된 오프셋 (-1.0 ~ 1.0)
        normalized_offset = offset / (image_width / 2)
        
        # 오프셋이 양수면 차선 중심이 오른쪽에 있으므로 왼쪽으로 조향 (음수)
        # 오프셋이 음수면 차선 중심이 왼쪽에 있으므로 오른쪽으로 조향 (양수)
        # 따라서 부호를 반대로 해야 함
        target_offset = -normalized_offset
        
        # 중앙선이 있으면 KP 증가 (더 강한 반응)
        from config import CENTER_LANE_KP_MULTIPLIER, CURVE_STEERING_MULTIPLIER
        effective_kp = self.kp * CENTER_LANE_KP_MULTIPLIER if has_center_lane else self.kp
        
        # 곡선 구간이면 조향 배율 적용
        if is_curve and curve_radius is not None:
            # 곡선 반경이 작을수록 (급커브) 더 강한 조향 필요
            # 반경이 작으면 배율 증가, 반경이 크면 배율 감소
            if curve_radius < image_width:  # 급커브
                curve_multiplier = CURVE_STEERING_MULTIPLIER * 1.5
            elif curve_radius < image_width * 2:  # 중간 커브
                curve_multiplier = CURVE_STEERING_MULTIPLIER
            else:  # 완만한 커브
                curve_multiplier = CURVE_STEERING_MULTIPLIER * 0.8
            
            effective_kp *= curve_multiplier
        
        # 비례 제어
        p_term = effective_kp * target_offset
        
        # 미분 제어
        d_term = self.kd * (target_offset - self.prev_offset)
        
        # 조향 각도 계산
        steering_angle = (p_term + d_term) * self.max_steering_angle
        
        # 각도 제한
        steering_angle = np.clip(steering_angle, 
                                -self.max_steering_angle, 
                                self.max_steering_angle)
        
        # 조향 각도 스무딩 (곡선 구간에서는 덜 스무딩하여 빠르게 반응)
        if is_curve:
            smoothing_factor = 0.4  # 곡선: 40% 유지 (더 빠른 반응)
        elif has_center_lane:
            smoothing_factor = 0.5  # 중앙선: 50% 유지
        else:
            smoothing_factor = 0.7  # 일반: 70% 유지
        
        steering_angle = (smoothing_factor * self.prev_steering_angle + 
                         (1 - smoothing_factor) * steering_angle)
        
        # 이전 값 업데이트
        self.prev_offset = target_offset
        self.prev_steering_angle = steering_angle
        
        return steering_angle
    
    def calculate_speed(self, offset: float, 
                       left_lane: Optional[np.ndarray] = None,
                       right_lane: Optional[np.ndarray] = None) -> float:
        """
        속도 계산
        
        Args:
            offset: 차선 중심으로부터의 오프셋
            left_lane: 왼쪽 차선 정보
            right_lane: 오른쪽 차선 정보
            
        Returns:
            speed: 속도 (0.0 ~ 1.0)
        """
        # 차선이 감지되지 않으면 감속
        if left_lane is None or right_lane is None:
            return self.min_speed * 0.5
        
        # 오프셋이 크면 감속 (차선 이탈 위험)
        abs_offset = abs(offset)
        if abs_offset > self.safe_offset_threshold:
            return self.min_speed
        
        # 정상 주행 시 속도 계산
        speed_factor = 1.0 - (abs_offset / self.safe_offset_threshold) * 0.3
        speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
        
        return np.clip(speed, self.min_speed, self.max_speed)
    
    def check_safety(self, offset: float, 
                    left_lane: Optional[np.ndarray] = None,
                    right_lane: Optional[np.ndarray] = None,
                    image_width: int = 640) -> Tuple[bool, str]:
        """
        안전성 검사
        
        Returns:
            (is_safe, message): 안전 여부와 메시지
        """
        # 차선 미감지
        if left_lane is None or right_lane is None:
            return False, "차선 미감지 - 정지 필요"
        
        # 차선 위치 확인
        left_x = left_lane[0][0]
        right_x = right_lane[0][0]
        center_x = (left_x + right_x) // 2
        image_center_x = image_width // 2
        
        # 노란색 선 바깥으로 나가는지 확인
        # 오프셋이 양수면 차선 중심이 오른쪽에 있음 (왼쪽으로 치우침)
        # 오프셋이 음수면 차선 중심이 왼쪽에 있음 (오른쪽으로 치우침)
        abs_offset = abs(offset)
        
        # 차선 이탈 위험 (임계값보다 크면)
        if abs_offset > self.safe_offset_threshold * 1.2:
            # 어느 쪽으로 치우쳤는지 확인
            if offset > 0:
                direction = "왼쪽"
            else:
                direction = "오른쪽"
            return False, f"차선 이탈 위험 ({direction}) - 오프셋: {abs_offset:.1f}px"
        
        # 차선 경계 확인 (노란색 선 위치)
        # 왼쪽 차선이 너무 오른쪽에 있거나, 오른쪽 차선이 너무 왼쪽에 있으면 위험
        lane_width = abs(right_x - left_x)
        expected_lane_width = image_width * 0.3  # 예상 차선 폭
        
        if lane_width < expected_lane_width * 0.5:
            return False, f"차선 폭 이상 - 너무 좁음: {lane_width:.1f}px"
        
        return True, "정상 주행"
    
    def get_control_command(self, 
                           offset: float,
                           left_lane: Optional[np.ndarray] = None,
                           right_lane: Optional[np.ndarray] = None,
                           center_lane: Optional[np.ndarray] = None,
                           image_width: int = 640,
                           is_curve: bool = False,
                           curve_radius: Optional[float] = None,
                           avoidance_command: Optional[Dict] = None,
                           aruco_command: Optional[Dict] = None) -> dict:
        """
        제어 명령 생성 (우선순위: 포트홀 회피 > ArUco 명령 > 차선 추종)
        
        Args:
            offset: 차선 중심으로부터의 오프셋
            left_lane: 왼쪽 차선 정보
            right_lane: 오른쪽 차선 정보
            center_lane: 중앙선 정보 (있으면 더 강하게 중앙 정렬)
            image_width: 이미지 너비
            is_curve: 곡선 구간 여부
            curve_radius: 곡선 반경 (픽셀)
            avoidance_command: 회피 명령 (최우선)
            aruco_command: ArUco 마커 명령 (2순위)
        
        Returns:
            control_command: 조향, 속도, 안전 정보를 포함한 딕셔너리
        """
        # 중앙선 감지 여부
        has_center_lane = center_lane is not None
        # 1순위: 포트홀 회피 명령 (최우선)
        if avoidance_command is not None and avoidance_command.get('action') != 'normal':
            target_offset = avoidance_command.get('target_offset', offset)
            avoidance_speed = avoidance_command.get('speed', self.min_speed)
            
            # 회피 목표 오프셋으로 조향 계산 (곡선 정보 포함)
            steering_angle = self.calculate_steering(target_offset, image_width, has_center_lane, 
                                                   is_curve, curve_radius)
            speed = avoidance_speed
            
            # 회피 상태 메시지
            action = avoidance_command.get('action', 'normal')
            if action.startswith('avoid_'):
                safety_message = f"포트홀 회피 중 ({action})"
            elif action == 'return':
                safety_message = "차선 복귀 중"
            else:
                safety_message = "정상 주행"
            
            is_safe = True  # 회피 중에는 안전하다고 가정
            
            return {
                'steering_angle': steering_angle,
                'speed': speed,
                'is_safe': is_safe,
                'safety_message': safety_message,
                'offset': target_offset,
                'avoidance_mode': True,
                'aruco_mode': False
            }
        
        # 2순위: ArUco 마커 명령
        if aruco_command is not None and aruco_command.get('action') != 'go_straight':
            action = aruco_command.get('action', 'go_straight')
            from config import ARUCO_TURN_ANGLE, ARUCO_TURN_SPEED
            
            if action == 'turn_right':
                steering_angle = ARUCO_TURN_ANGLE  # 우회전
                safety_message = "ArUco: 우회전"
            elif action == 'turn_left':
                steering_angle = -ARUCO_TURN_ANGLE  # 좌회전
                safety_message = "ArUco: 좌회전"
            elif action == 'stop':
                steering_angle = 0.0
                safety_message = "ArUco: 정지"
                return {
                    'steering_angle': 0.0,
                    'speed': 0.0,
                    'is_safe': False,
                    'safety_message': safety_message,
                    'offset': offset,
                    'avoidance_mode': False,
                    'aruco_mode': True
                }
            else:
                # go_straight는 차선 추종으로 처리 (곡선 정보 포함)
                steering_angle = self.calculate_steering(offset, image_width, has_center_lane, 
                                                        is_curve, curve_radius)
                safety_message = "ArUco: 직진 (차선 추종)"
            
            return {
                'steering_angle': steering_angle,
                'speed': ARUCO_TURN_SPEED,
                'is_safe': True,
                'safety_message': safety_message,
                'offset': offset,
                'avoidance_mode': False,
                'aruco_mode': True
            }
        
        # 정상 주행 모드
        # 중앙선이 있으면 더 강하게 중앙 정렬 (곡선 정보 포함)
        steering_angle = self.calculate_steering(offset, image_width, has_center_lane, 
                                                is_curve, curve_radius)
        speed = self.calculate_speed(offset, left_lane, right_lane)
        is_safe, safety_message = self.check_safety(offset, left_lane, right_lane, image_width)
        
        # 차선 이탈 방지: 안전하지 않으면 강제 조정
        if not is_safe and abs(offset) > self.safe_offset_threshold:
            # 차선 중심으로 강제 복귀 (중앙선이 있으면 더 강하게, 곡선 정보 포함)
            correction_factor = 0.7 if has_center_lane else 0.5  # 중앙선: 70% 보정, 일반: 50% 보정
            correction_offset = -offset * correction_factor
            steering_angle = self.calculate_steering(correction_offset, image_width, has_center_lane,
                                                    is_curve, curve_radius)
            speed = self.min_speed  # 감속
        
        return {
            'steering_angle': steering_angle,
            'speed': speed,
            'is_safe': is_safe,
            'safety_message': safety_message,
            'offset': offset,
            'avoidance_mode': False
        }
    
    def steering_to_motor_speed(self, steering_angle: float, speed: float) -> Tuple[int, int]:
        """
        조향 각도와 속도를 좌우 모터 속도로 변환
        
        Args:
            steering_angle: 조향 각도 (도, -max ~ +max)
            speed: 속도 (0.0 ~ 1.0)
            
        Returns:
            (left_speed, right_speed): 좌우 모터 속도 (-127 ~ 127)
        """
        # 기본 속도 계산 (0.0 ~ 1.0 -> -127 ~ 127)
        base_speed = int(speed * self.base_speed_range)
        
        # 조향 각도를 속도 차이로 변환
        # steering_angle이 양수면 오른쪽 회전 (왼쪽 모터 빠르게)
        # steering_angle이 음수면 왼쪽 회전 (오른쪽 모터 빠르게)
        steering_factor = steering_angle / self.max_steering_angle  # -1.0 ~ 1.0
        speed_diff = int(steering_factor * self.turn_sensitivity * self.base_speed_range)
        
        # 좌우 모터 속도 계산
        left_speed = base_speed + speed_diff
        right_speed = base_speed - speed_diff
        
        # 속도 제한 (-127 ~ 127)
        left_speed = np.clip(left_speed, -self.base_speed_range, self.base_speed_range)
        right_speed = np.clip(right_speed, -self.base_speed_range, self.base_speed_range)
        
        return left_speed, right_speed
    
    def execute_motor_control(self, steering_angle: float, speed: float):
        """
        모터 제어 실행 (TikiMini API 사용)
        
        Args:
            steering_angle: 조향 각도 (도)
            speed: 속도 (0.0 ~ 1.0)
        """
        if not self.use_tiki or self.tiki is None:
            return
        
        try:
            # 조향 각도와 속도를 모터 속도로 변환
            left_speed, right_speed = self.steering_to_motor_speed(steering_angle, speed)
            
            # TikiMini API로 모터 제어
            self.tiki.set_motor_power(self.tiki.MOTOR_LEFT, left_speed)
            self.tiki.set_motor_power(self.tiki.MOTOR_RIGHT, right_speed)
            
        except Exception as e:
            print(f"모터 제어 오류: {e}")
    
    def stop_motors(self):
        """모터 정지"""
        if self.use_tiki and self.tiki is not None:
            try:
                self.tiki.stop()
            except Exception as e:
                print(f"모터 정지 오류: {e}")

