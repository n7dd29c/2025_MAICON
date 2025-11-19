"""
차선 인식 모듈
Jetson Nano 최적화된 차선 감지 시스템
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


class LaneDetector:
    """차선 인식 클래스 - 흰색(중앙) 및 노란색(양쪽) 차선 감지"""
    
    def __init__(self, 
                 image_width: int = 640,
                 image_height: int = 480,
                 roi_ratio: float = 0.6):
        """
        Args:
            image_width: 처리할 이미지 너비
            image_height: 처리할 이미지 높이
            roi_ratio: 관심 영역(ROI) 비율 (하단 부분만 사용)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.roi_ratio = roi_ratio
        
        # ROI 영역 계산 (하단 부분만 사용)
        self.roi_top = int(image_height * (1 - roi_ratio))
        self.roi_bottom = image_height
        
        # 차선 색상 범위 (HSV)
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        
        # 이전 차선 정보 (안정화를 위해)
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.smoothing_factor = 0.7
        
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 리사이즈 (성능 향상)
        frame = cv2.resize(frame, (self.image_width, self.image_height))
        
        # ROI 추출 (하단 부분만)
        roi = frame[self.roi_top:self.roi_bottom, :]
        
        return roi
    
    def detect_lanes(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        차선 감지
        
        Returns:
            (left_lane, right_lane): 왼쪽/오른쪽 차선 좌표 (None if not detected)
        """
        # 전처리
        roi = self.preprocess_image(frame)
        
        # HSV 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 흰색 차선 마스크
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # 노란색 차선 마스크
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 마스크 결합
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 마스크 적용
        masked_edges = cv2.bitwise_and(edges, lane_mask)
        
        # Hough 변환으로 직선 검출
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=100
        )
        
        if lines is None:
            return self.prev_left_lane, self.prev_right_lane
        
        # 차선 분류 (왼쪽/오른쪽)
        left_lines, right_lines = self._classify_lanes(lines)
        
        # 차선 평균화
        left_lane = self._average_lane(left_lines) if left_lines else None
        right_lane = self._average_lane(right_lines) if right_lines else None
        
        # 이전 값과 스무딩
        if left_lane is not None:
            if self.prev_left_lane is not None:
                left_lane = self._smooth_lane(left_lane, self.prev_left_lane)
            self.prev_left_lane = left_lane
        else:
            left_lane = self.prev_left_lane
        
        if right_lane is not None:
            if self.prev_right_lane is not None:
                right_lane = self._smooth_lane(right_lane, self.prev_right_lane)
            self.prev_right_lane = right_lane
        else:
            right_lane = self.prev_right_lane
        
        return left_lane, right_lane
    
    def _classify_lanes(self, lines: np.ndarray) -> Tuple[List, List]:
        """차선을 왼쪽/오른쪽으로 분류"""
        left_lines = []
        right_lines = []
        
        mid_x = self.image_width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 수평선 제외
            if abs(y2 - y1) < 10:
                continue
            
            # 기울기 계산
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            
            # 기울기 필터링 (너무 수평이거나 수직인 선 제외)
            if abs(slope) < 0.3 or abs(slope) > 3:
                continue
            
            # 왼쪽/오른쪽 분류
            if slope < 0:  # 왼쪽 차선 (음의 기울기)
                left_lines.append(line[0])
            else:  # 오른쪽 차선 (양의 기울기)
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def _average_lane(self, lines: List) -> Optional[np.ndarray]:
        """여러 선을 평균화하여 하나의 차선으로 만듦"""
        if not lines:
            return None
        
        # 기울기와 절편 계산
        slopes = []
        intercepts = []
        
        for x1, y1, x2, y2 in lines:
            if (x2 - x1) == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            slopes.append(slope)
            intercepts.append(intercept)
        
        if not slopes:
            return None
        
        # 평균 계산
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        
        # 차선의 시작점과 끝점 계산
        y1 = self.roi_bottom - self.roi_top
        y2 = int(y1 * 0.6)
        
        x1 = int((y1 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
        x2 = int((y2 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
        
        # 이미지 경계 내로 제한
        x1 = max(0, min(self.image_width - 1, x1))
        x2 = max(0, min(self.image_width - 1, x2))
        
        return np.array([[x1, y1, x2, y2]], dtype=np.int32)
    
    def _smooth_lane(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """이전 차선 정보와 스무딩"""
        if previous is None:
            return current
        
        smoothed = current.copy()
        smoothed[0] = (self.smoothing_factor * previous[0] + 
                      (1 - self.smoothing_factor) * current[0]).astype(np.int32)
        
        return smoothed
    
    def calculate_center(self, left_lane: Optional[np.ndarray], 
                        right_lane: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
        """
        차선 중심점 계산
        
        Returns:
            (center_x, center_y): 중심점 좌표
        """
        if left_lane is None or right_lane is None:
            return None
        
        # 하단 부분의 중심점 계산
        left_x = left_lane[0][0]
        right_x = right_lane[0][0]
        
        center_x = (left_x + right_x) // 2
        center_y = left_lane[0][1]
        
        return (center_x, center_y)
    
    def calculate_offset(self, center_point: Optional[Tuple[int, int]]) -> float:
        """
        차선 중심으로부터의 오프셋 계산
        
        Returns:
            offset: 픽셀 단위 오프셋 (음수: 왼쪽, 양수: 오른쪽)
        """
        if center_point is None:
            return 0.0
        
        image_center_x = self.image_width // 2
        offset = center_point[0] - image_center_x
        
        return offset
    
    def draw_lanes(self, frame: np.ndarray, 
                   left_lane: Optional[np.ndarray],
                   right_lane: Optional[np.ndarray],
                   center_point: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """차선 시각화"""
        vis_frame = frame.copy()
        
        # ROI 영역 표시
        cv2.rectangle(vis_frame, 
                     (0, self.roi_top), 
                     (self.image_width, self.roi_bottom),
                     (0, 255, 0), 2)
        
        # 왼쪽 차선 그리기
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane[0]
            cv2.line(vis_frame, 
                    (x1, y1 + self.roi_top), 
                    (x2, y2 + self.roi_top),
                    (255, 0, 0), 3)
        
        # 오른쪽 차선 그리기
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane[0]
            cv2.line(vis_frame, 
                    (x1, y1 + self.roi_top), 
                    (x2, y2 + self.roi_top),
                    (0, 0, 255), 3)
        
        # 중심점 표시
        if center_point is not None:
            cx, cy = center_point
            cv2.circle(vis_frame, 
                      (cx, cy + self.roi_top), 
                      5, (0, 255, 0), -1)
            
            # 이미지 중심선
            image_center_x = self.image_width // 2
            cv2.line(vis_frame,
                    (image_center_x, self.roi_top),
                    (image_center_x, self.roi_bottom),
                    (255, 255, 0), 2)
        
        return vis_frame

