"""
자율주행 제어 모듈
차선 인식 결과를 바탕으로 조향 및 속도 제어
"""
import numpy as np
from typing import Optional, Tuple
from lane_detector import LaneDetector


class AutonomousController:
    """자율주행 제어 클래스"""
    
    def __init__(self,
                 max_steering_angle: float = 30.0,
                 kp: float = 0.5,
                 kd: float = 0.1,
                 max_speed: float = 0.5,
                 min_speed: float = 0.2):
        """
        Args:
            max_steering_angle: 최대 조향 각도 (도)
            kp: 비례 제어 게인
            kd: 미분 제어 게인
            max_speed: 최대 속도 (0.0 ~ 1.0)
            min_speed: 최소 속도 (0.0 ~ 1.0)
        """
        self.max_steering_angle = max_steering_angle
        self.kp = kp
        self.kd = kd
        self.max_speed = max_speed
        self.min_speed = min_speed
        
        # 이전 오프셋 (미분 제어용)
        self.prev_offset = 0.0
        
        # 안전 임계값
        self.safe_offset_threshold = 50  # 픽셀 단위
        
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
                           image_width: int = 640) -> dict:
        """
        제어 명령 생성
        
        Returns:
            control_command: 조향, 속도, 안전 정보를 포함한 딕셔너리
        """
        steering_angle = self.calculate_steering(offset, image_width)
        speed = self.calculate_speed(offset, left_lane, right_lane)
        is_safe, safety_message = self.check_safety(offset, left_lane, right_lane)
        
        return {
            'steering_angle': steering_angle,
            'speed': speed,
            'is_safe': is_safe,
            'safety_message': safety_message,
            'offset': offset
        }

