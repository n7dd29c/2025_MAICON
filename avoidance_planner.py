"""
회피 경로 계획 모듈
포트홀 회피를 위한 경로 계획
"""
import numpy as np
from typing import Optional, Dict, Tuple
from lane_detector import LaneDetector


class AvoidancePlanner:
    """회피 경로 계획 클래스"""
    
    def __init__(self,
                 image_width: int = 640,
                 image_height: int = 480):
        """
        Args:
            image_width: 이미지 너비
            image_height: 이미지 높이
        """
        self.image_width = image_width
        self.image_height = image_height
        
        # 회피 상태
        self.avoidance_state = "normal"  # normal, avoiding, returning
        self.target_pothole = None
        self.avoidance_direction = None  # "left" or "right"
        self.avoidance_start_offset = 0.0
        
        # 회피 파라미터
        self.avoidance_margin = 50  # 포트홀으로부터의 안전 거리 (픽셀)
        self.return_threshold = 30  # 복귀 임계값 (픽셀)
        
    def plan_avoidance(self,
                      pothole: Dict,
                      left_lane: Optional[np.ndarray],
                      right_lane: Optional[np.ndarray],
                      current_offset: float) -> Dict:
        """
        회피 경로 계획
        
        Args:
            pothole: 포트홀 정보
            left_lane: 왼쪽 차선
            right_lane: 오른쪽 차선
            current_offset: 현재 차선 중심으로부터의 오프셋
            
        Returns:
            avoidance_command: 회피 명령 {'action': 'avoid_left/avoid_right/normal', 
                                         'target_offset': float, 'speed': float}
        """
        if left_lane is None or right_lane is None:
            # 차선이 없으면 정상 주행
            self.avoidance_state = "normal"
            return {
                'action': 'normal',
                'target_offset': current_offset,
                'speed': 0.5
            }
        
        pothole_x, pothole_y = pothole['center']
        pothole_width = pothole['bbox'][2]
        
        # 차선 위치 계산
        left_x = left_lane[0][0]
        right_x = right_lane[0][0]
        lane_center_x = (left_x + right_x) // 2
        lane_width = right_x - left_x
        
        # 포트홀의 상대적 위치
        pothole_offset = pothole_x - lane_center_x
        
        # 회피 방향 결정
        # 포트홀이 차선 중앙에 있으면, 더 넓은 쪽으로 회피
        if abs(pothole_offset) < lane_width * 0.2:
            # 양쪽 차선의 여유 공간 확인
            left_space = left_x
            right_space = self.image_width - right_x
            
            if left_space > right_space:
                avoidance_direction = "left"
                target_offset = -lane_width * 0.6  # 왼쪽으로 크게 벗어남
            else:
                avoidance_direction = "right"
                target_offset = lane_width * 0.6  # 오른쪽으로 크게 벗어남
        else:
            # 포트홀 반대 방향으로 회피
            if pothole_offset > 0:  # 포트홀이 오른쪽에 있음
                avoidance_direction = "left"
                target_offset = -lane_width * 0.6
            else:  # 포트홀이 왼쪽에 있음
                avoidance_direction = "right"
                target_offset = lane_width * 0.6
        
        # 회피 상태 업데이트
        self.avoidance_state = "avoiding"
        self.target_pothole = pothole
        self.avoidance_direction = avoidance_direction
        self.avoidance_start_offset = current_offset
        
        return {
            'action': f'avoid_{avoidance_direction}',
            'target_offset': target_offset,
            'speed': 0.3,  # 회피 시 감속
            'pothole': pothole
        }
    
    def check_return_condition(self,
                              current_offset: float,
                              left_lane: Optional[np.ndarray],
                              right_lane: Optional[np.ndarray]) -> bool:
        """
        차선 복귀 조건 확인
        
        Args:
            current_offset: 현재 오프셋
            left_lane: 왼쪽 차선
            right_lane: 오른쪽 차선
            
        Returns:
            should_return: 복귀해야 하는지 여부
        """
        if self.avoidance_state != "avoiding":
            return False
        
        if left_lane is None or right_lane is None:
            return False
        
        # 포트홀이 더 이상 감지되지 않거나, 충분히 벗어났으면 복귀
        if self.target_pothole is None:
            return True
        
        # 현재 위치가 포트홀을 지나쳤는지 확인
        pothole_x = self.target_pothole['center'][0]
        current_x = self.image_width // 2 + current_offset
        
        # 포트홀을 지나쳤고, 충분히 벗어났으면 복귀 시작
        if self.avoidance_direction == "left":
            # 왼쪽으로 회피 중이면, 포트홀보다 왼쪽에 있고 충분히 벗어났는지 확인
            if current_x < pothole_x - self.avoidance_margin:
                return True
        else:  # right
            # 오른쪽으로 회피 중이면, 포트홀보다 오른쪽에 있고 충분히 벗어났는지 확인
            if current_x > pothole_x + self.avoidance_margin:
                return True
        
        return False
    
    def plan_return(self,
                   current_offset: float,
                   left_lane: Optional[np.ndarray],
                   right_lane: Optional[np.ndarray]) -> Dict:
        """
        차선 복귀 경로 계획
        
        Args:
            current_offset: 현재 오프셋
            left_lane: 왼쪽 차선
            right_lane: 오른쪽 차선
            
        Returns:
            return_command: 복귀 명령
        """
        if left_lane is None or right_lane is None:
            self.avoidance_state = "normal"
            return {
                'action': 'normal',
                'target_offset': 0.0,
                'speed': 0.5
            }
        
        # 차선 중심으로 복귀
        target_offset = 0.0
        
        # 복귀 상태 업데이트
        self.avoidance_state = "returning"
        
        # 복귀 완료 확인
        if abs(current_offset) < self.return_threshold:
            self.avoidance_state = "normal"
            self.target_pothole = None
            self.avoidance_direction = None
        
        return {
            'action': 'return',
            'target_offset': target_offset,
            'speed': 0.4  # 복귀 시 중간 속도
        }
    
    def reset(self):
        """회피 상태 초기화"""
        self.avoidance_state = "normal"
        self.target_pothole = None
        self.avoidance_direction = None
        self.avoidance_start_offset = 0.0

