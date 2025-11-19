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
        
        # 안전 임계값
        self.safe_offset_threshold = 50  # 픽셀 단위
        
        # 회피 명령 (외부에서 설정)
        self.avoidance_command: Optional[Dict] = None
        
        # 모터 제어 파라미터
        self.base_speed_range = 127  # TikiMini 모터 속도 범위 (-127 ~ 127)
        from config import TURN_SENSITIVITY
        self.turn_sensitivity = TURN_SENSITIVITY  # 조향 민감도
        
    def calculate_steering(self, offset: float, image_width: int = 640) -> float:
        """
        조향 각도 계산 (PID 제어)
        
        Args:
            offset: 차선 중심으로부터의 오프셋 (픽셀)
            image_width: 이미지 너비
            
        Returns:
            steering_angle: 조향 각도 (-max_steering_angle ~ max_steering_angle)
        """
        # 정규화된 오프셋 (-1.0 ~ 1.0)
        normalized_offset = offset / (image_width / 2)
        
        # 비례 제어
        p_term = self.kp * normalized_offset
        
        # 미분 제어
        d_term = self.kd * (normalized_offset - self.prev_offset)
        
        # 조향 각도 계산
        steering_angle = (p_term + d_term) * self.max_steering_angle
        
        # 각도 제한
        steering_angle = np.clip(steering_angle, 
                                -self.max_steering_angle, 
                                self.max_steering_angle)
        
        # 이전 값 업데이트
        self.prev_offset = normalized_offset
        
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
                    right_lane: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        안전성 검사
        
        Returns:
            (is_safe, message): 안전 여부와 메시지
        """
        # 차선 미감지
        if left_lane is None or right_lane is None:
            return False, "차선 미감지 - 정지 필요"
        
        # 노란선 바깥으로 나가는지 확인 (오프셋이 너무 큰 경우)
        abs_offset = abs(offset)
        if abs_offset > self.safe_offset_threshold * 1.5:
            return False, f"차선 이탈 위험 - 오프셋: {abs_offset:.1f}px"
        
        return True, "정상 주행"
    
    def get_control_command(self, 
                           offset: float,
                           left_lane: Optional[np.ndarray] = None,
                           right_lane: Optional[np.ndarray] = None,
                           image_width: int = 640,
                           avoidance_command: Optional[Dict] = None) -> dict:
        """
        제어 명령 생성
        
        Args:
            offset: 차선 중심으로부터의 오프셋
            left_lane: 왼쪽 차선 정보
            right_lane: 오른쪽 차선 정보
            image_width: 이미지 너비
            avoidance_command: 회피 명령 (있으면 우선 적용)
        
        Returns:
            control_command: 조향, 속도, 안전 정보를 포함한 딕셔너리
        """
        # 회피 명령이 있으면 우선 적용
        if avoidance_command is not None and avoidance_command.get('action') != 'normal':
            target_offset = avoidance_command.get('target_offset', offset)
            avoidance_speed = avoidance_command.get('speed', self.min_speed)
            
            # 회피 목표 오프셋으로 조향 계산
            steering_angle = self.calculate_steering(target_offset, image_width)
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
                'avoidance_mode': True
            }
        
        # 정상 주행 모드
        steering_angle = self.calculate_steering(offset, image_width)
        speed = self.calculate_speed(offset, left_lane, right_lane)
        is_safe, safety_message = self.check_safety(offset, left_lane, right_lane)
        
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

