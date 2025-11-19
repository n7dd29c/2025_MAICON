"""
차선 인식 모듈
Jetson Nano 최적화된 차선 감지 시스템
Bird's Eye View 적용
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from perspective_transform import PerspectiveTransform


class LaneDetector:
    """차선 인식 클래스 - 흰색(중앙) 및 노란색(양쪽) 차선 감지"""
    
    def __init__(self, 
                 image_width: int = 640,
                 image_height: int = 480,
                 roi_ratio: float = 0.6,
                 use_bird_view: bool = True):
        """
        Args:
            image_width: 처리할 이미지 너비
            image_height: 처리할 이미지 높이
            roi_ratio: 관심 영역(ROI) 비율 (하단 부분만 사용)
            use_bird_view: Bird's Eye View 사용 여부
        """
        self.image_width = image_width
        self.image_height = image_height
        self.roi_ratio = roi_ratio
        self.use_bird_view = use_bird_view
        
        # ROI 영역 계산 (하단 부분만 사용)
        self.roi_top = int(image_height * (1 - roi_ratio))
        self.roi_bottom = image_height
        
        # Bird's Eye View 변환기 초기화
        if self.use_bird_view:
            self.perspective_transform = PerspectiveTransform(
                image_width=image_width,
                image_height=image_height,
                roi_ratio=roi_ratio
            )
            # 변환된 이미지 크기
            self.warped_width, self.warped_height = self.perspective_transform.get_warped_size()
        else:
            self.perspective_transform = None
            self.warped_width = image_width
            self.warped_height = int(image_height * roi_ratio)
        
        # 차선 색상 범위 (HSV)
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        
        # 이전 차선 정보 (안정화를 위해)
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.prev_center_lane = None
        self.smoothing_factor = 0.8  # 0.7 -> 0.8 (더 강한 스무딩)
        
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """이미지 전처리 (Bird's Eye View 변환 포함)"""
        # 리사이즈 (성능 향상)
        frame = cv2.resize(frame, (self.image_width, self.image_height))
        
        # Bird's Eye View 변환
        if self.use_bird_view and self.perspective_transform:
            warped = self.perspective_transform.warp(frame)
            return warped
        
        # ROI 추출 (하단 부분만) - 변환 없이 사용
        roi = frame[self.roi_top:self.roi_bottom, :]
        return roi
    
    def detect_lanes(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        차선 감지 (색상 기반, Hough 변환 없이)
        shape[:2]
        else:
            height, width = roi.shape[:2]
        
        # HSV 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 흰색 차선 마스크 (중앙선)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # 노란색 차선 마스크
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # 색상 기반 차선 감지 (히스토그램 방식)
        # 하단 부분에서 차선 위치 찾기
        bottom_half = height // 2  # 하단 절반만 사용
        
        # 중앙선 감지 (흰색 마스크)
        center_lane = self._detect_lane_from_mask(white_mask, bottom_half, height, width, is_center=True)
        
        # 왼쪽 차선 감지 (노란색 마스크의 왼쪽 영역)
        left_yellow = yellow_mask[:, :width//2]  # 왼쪽 절반만
        left_lane = self._detect_lane_from_mask(left_yellow, bottom_half, height, width//2, is_left=True)
        # 원본 좌표로 변환 (왼쪽 절반이므로 x 좌표 그대로 사용 - 이미 올바름)
        
        # 오른쪽 차선 감지 (노란색 마스크의 오른쪽 영역)
        right_yellow = yellow_mask[:, width//2:]  # 오른쪽 절반만
        right_lane = self._detect_lane_from_mask(right_yellow, bottom_half, height, width//2, is_left=False)
        if right_lane is not None:
            # 원본 좌표로 변환 (오른쪽 절반이므로 x 좌표에 width//2 더하기)
            right_lane[0][0] += width // 2
            right_lane[0][2] += width // 2
        
        # 이전 값과 스무딩
        if left_lane is not None:
            if self.prev_left_lane is not None:
                left_lane = self._smooth_lane(left_lane, self.prev_left_lane)
            self.prev_left_lane = left_lane
        else:
            # 차선이 감지되지 않으면 이전 값 사용 (안정성)
            left_lane = self.prev_left_lane
        
        if right_lane is not None:
            if self.prev_right_lane is not None:
                right_lane = self._smooth_lane(right_lane, self.prev_right_lane)
            self.prev_right_lane = right_lane
        else:
            # 차선이 감지되지 않으면 이전 값 사용 (안정성)
            right_lane = self.prev_right_lane
        
        # 중앙선 스무딩
        if center_lane is not None:
            if self.prev_center_lane is not None:
                center_lane = self._smooth_lane(center_lane, self.prev_center_lane)
            self.prev_center_lane = center_lane
        else:
            # 중앙선이 감지되지 않으면 이전 값 사용
            center_lane = self.prev_center_lane
        
        # 차선 유효성 검사 (너무 가까이 있거나 이상한 경우 필터링)
        if left_lane is not None and right_lane is not None:
            left_x = left_lane[0][0]
            right_x = right_lane[0][0]
            
            # 이미지 크기 확인
            if self.use_bird_view:
                width_limit = self.warped_width
            else:
                width_limit = self.image_width
            
            # 차선 간격이 너무 좁으면 무효
            lane_width = abs(right_x - left_x)
            min_lane_width = width_limit * 0.2  # 최소 차선 폭
            if lane_width < min_lane_width:
                # 이전 값 사용 (3개 모두 반환)
                return self.prev_left_lane, self.prev_right_lane, self.prev_center_lane
            
            # 차선이 교차하거나 순서가 바뀐 경우 필터링
            if left_x >= right_x:
                # 이전 값 사용
                return self.prev_left_lane, self.prev_right_lane, self.prev_center_lane
        
        return left_lane, right_lane, center_lane
    
    def _detect_lane_from_mask(self, mask: np.ndarray, bottom_half: int, height: int, width: int, 
                               is_center: bool = False, is_left: bool = False) -> Optional[np.ndarray]:
        """
        마스크에서 색상 기반으로 차선 위치 찾기 (히스토그램 방식)
        
        Args:
            mask: 차선 마스크 (흰색 또는 노란색)
            bottom_half: 하단 절반 시작 y 좌표
            height: 이미지 높이
            width: 이미지 너비
            is_center: 중앙선인지 여부
            is_left: 왼쪽 차선인지 여부
        
        Returns:
            lane: 차선 좌표 [x1, y1, x2, y2] 또는 None
        """
        if mask is None or mask.size == 0:
            return None
        
        # 마스크 실제 크기
        mask_height, mask_width = mask.shape[:2]
        
        # 각 행에서 차선 픽셀의 x 좌표 평균 계산
        lane_points = []  # [(x, y), ...]
        
        # 하단부터 상단으로 스캔 (슬라이딩 윈도우)
        window_height = max(5, mask_height // 10)  # 윈도우 높이 (최소 5픽셀)
        step_size = max(2, window_height // 2)  # 스텝 크기
        
        # 하단 영역에서 시작 (마스크의 하단 절반)
        mask_bottom_half = mask_height // 2
        
        for y_offset in range(0, mask_height - mask_bottom_half - window_height, step_size):
            y = mask_bottom_half + y_offset
            if y + window_height > mask_height:
                break
                
            window = mask[y:y+window_height, :]
            
            # 이 윈도우에서 차선 픽셀의 x 좌표들
            # 각 열에서 차선 픽셀이 있는지 확인
            column_sums = np.sum(window, axis=0)
            lane_pixels_x = np.where(column_sums > 0)[0]
            
            if len(lane_pixels_x) > 0:
                # 차선 픽셀의 중심 x 좌표 (가중 평균)
                # 더 많은 픽셀이 있는 열에 더 높은 가중치
                weights = column_sums[lane_pixels_x]
                center_x = int(np.average(lane_pixels_x, weights=weights))
                center_y = y + window_height // 2
                lane_points.append((center_x, center_y))
        
        if len(lane_points) < 2:
            return None
        
        # 차선 포인트들을 직선으로 피팅
        lane_points = np.array(lane_points)
        x_coords = lane_points[:, 0]
        y_coords = lane_points[:, 1]
        
        # 최소 제곱법으로 직선 피팅 (x = my + b 형태)
        if len(x_coords) > 1:
            coeffs = np.polyfit(y_coords, x_coords, 1)
            slope = coeffs[0]  # m
            intercept = coeffs[1]  # b
            
            # 하단과 상단의 x 좌표 계산
            y1 = mask_height - 1  # 하단
            y2 = mask_bottom_half  # 상단
            
            x1 = int(slope * y1 + intercept)
            x2 = int(slope * y2 + intercept)
            
            # 이미지 경계 내로 제한
            x1 = max(0, min(mask_width - 1, x1))
            x2 = max(0, min(mask_width - 1, x2))
            
            return np.array([[x1, y1, x2, y2]], dtype=np.int32)
        
        return None
    
    # ============================================================
    # 레거시 코드: Hough 변환 방식 (현재 사용 안 함, 색상 기반 방식 사용)
    # ============================================================
    # 
    # def _classify_lanes(self, lines: np.ndarray, white_mask: np.ndarray) -> Tuple[List, List, List]:
    #     """차선을 왼쪽/오른쪽/중앙으로 분류 (기울기 + 위치 기반)"""
    #     left_lines = []
    #     right_lines = []
    #     center_lines = []
    #     
    #     if self.use_bird_view:
    #         mid_x = self.warped_width // 2
    #         width_limit = self.warped_width
    #     else:
    #         mid_x = self.image_width // 2
    #         width_limit = self.image_width
    #     
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         
    #         # 수평선 제외
    #         if abs(y2 - y1) < 10:
    #             continue
    #         
    #         # 기울기 계산
    #         slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
    #         
    #         # 기울기 필터링 (너무 수평이거나 수직인 선 제외)
    #         if abs(slope) < 0.2 or abs(slope) > 5:
    #             continue
    #         
    #         # 위치 기반 분류 (하단 부분의 x 좌표 기준)
    #         if y2 > y1:
    #             bottom_x = x2
    #         else:
    #             bottom_x = x1
    #         
    #         # 중앙선 판단: 중앙 영역에 있고 기울기가 작은 선 (하얀색 선)
    #         # 중앙 영역: 이미지 중앙 ± 30%
    #         center_zone_left = mid_x * 0.7
    #         center_zone_right = mid_x * 1.3
    #         
    #         # 하얀색 마스크에서 선이 지나가는 부분 확인
    #         line_mid_x = (x1 + x2) // 2
    #         line_mid_y = (y1 + y2) // 2
    #         is_white = False
    #         if 0 <= line_mid_x < width_limit and 0 <= line_mid_y < white_mask.shape[0]:
    #             is_white = white_mask[line_mid_y, line_mid_x] > 0
    #         
    #         # 중앙선: 중앙 영역에 있고 기울기가 작으며 하얀색인 선
    #         if (center_zone_left <= bottom_x <= center_zone_right and 
    #             abs(slope) < 0.5 and is_white):
    #             center_lines.append(line[0])
    #         # 왼쪽 차선: 음의 기울기 또는 중앙보다 왼쪽에 위치
    #         elif slope < -0.2 or (slope < 0.2 and bottom_x < center_zone_left):
    #             left_lines.append(line[0])
    #         # 오른쪽 차선: 양의 기울기 또는 중앙보다 오른쪽에 위치
    #         elif slope > 0.2 or (slope > -0.2 and bottom_x > center_zone_right):
    #             right_lines.append(line[0])
    #     
    #     return left_lines, right_lines, center_lines
    # 
    # def _average_lane(self, lines: List) -> Optional[np.ndarray]:
    #     """여러 선을 평균화하여 하나의 차선으로 만듦"""
    #     if not lines:
    #         return None
    #     
    #     # 기울기와 절편 계산
    #     slopes = []
    #     intercepts = []
    #     
    #     for x1, y1, x2, y2 in lines:
    #         if (x2 - x1) == 0:
    #             continue
    #         
    #         slope = (y2 - y1) / (x2 - x1)
    #         intercept = y1 - slope * x1
    #         
    #         slopes.append(slope)
    #         intercepts.append(intercept)
    #     
    #     if not slopes:
    #         return None
    #     
    #     # 평균 계산
    #     avg_slope = np.mean(slopes)
    #     avg_intercept = np.mean(intercepts)
    #     
    #     # 차선의 시작점과 끝점 계산
    #     if self.use_bird_view:
    #         # Bird's Eye View에서는 변환된 이미지 크기 사용
    #         y1 = self.warped_height
    #         y2 = int(y1 * 0.6)
    #         width_limit = self.warped_width
    #     else:
    #         y1 = self.roi_bottom - self.roi_top
    #         y2 = int(y1 * 0.6)
    #         width_limit = self.image_width
    #     
    #     x1 = int((y1 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
    #     x2 = int((y2 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
    #     
    #     # 이미지 경계 내로 제한
    #     x1 = max(0, min(width_limit - 1, x1))
    #     x2 = max(0, min(width_limit - 1, x2))
    #     
    #     return np.array([[x1, y1, x2, y2]], dtype=np.int32)
    
    def _smooth_lane(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """이전 차선 정보와 스무딩"""
        if previous is None:
            return current
        
        smoothed = current.copy()
        smoothed[0] = (self.smoothing_factor * previous[0] + 
                      (1 - self.smoothing_factor) * current[0]).astype(np.int32)
        
        return smoothed
    
    def calculate_center(self, left_lane: Optional[np.ndarray], 
                        right_lane: Optional[np.ndarray],
                        center_lane: Optional[np.ndarray] = None) -> Optional[Tuple[int, int]]:
        """
        차선 중심점 계산 (중앙선 우선)
        
        Returns:
            (center_x, center_y): 중심점 좌표
        """
        # 중앙선이 있으면 중앙선 기준
        if center_lane is not None:
            center_x = center_lane[0][0]
            center_y = center_lane[0][1]
            return (center_x, center_y)
        
        # 중앙선이 없으면 왼쪽/오른쪽 차선의 중심
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
        
        # Bird's Eye View 사용 시 변환된 이미지 기준
        if self.use_bird_view:
            image_center_x = self.warped_width // 2
        else:
            image_center_x = self.image_width // 2
        
        offset = center_point[0] - image_center_x
        
        return offset
    
    def draw_lanes(self, frame: np.ndarray, 
                   left_lane: Optional[np.ndarray],
                   right_lane: Optional[np.ndarray],
                   center_point: Optional[Tuple[int, int]] = None,
                   center_lane: Optional[np.ndarray] = None) -> np.ndarray:
        """차선 시각화 (중앙선 포함)"""
        vis_frame = frame.copy()
        
        if self.use_bird_view and self.perspective_transform:
            # Bird's Eye View에서 차선 그리기
            # 변환된 이미지에 그리기
            warped_vis = self.perspective_transform.warp(vis_frame)
            
            # 중앙선 그리기 (하얀색, 두껍게) - 우선 표시
            if center_lane is not None:
                x1, y1, x2, y2 = center_lane[0]
                cv2.line(warped_vis, 
                        (x1, y1), 
                        (x2, y2),
                        (255, 255, 255), 4)  # 하얀색, 두껍게
            
            # 왼쪽 차선 그리기 (노란색)
            if left_lane is not None:
                x1, y1, x2, y2 = left_lane[0]
                cv2.line(warped_vis, 
                        (x1, y1), 
                        (x2, y2),
                        (0, 255, 255), 3)  # 노란색 (BGR)
            
            # 오른쪽 차선 그리기 (노란색)
            if right_lane is not None:
                x1, y1, x2, y2 = right_lane[0]
                cv2.line(warped_vis, 
                        (x1, y1), 
                        (x2, y2),
                        (0, 255, 255), 3)  # 노란색 (BGR)
            
            # 중심점 표시
            if center_point is not None:
                cx, cy = center_point
                cv2.circle(warped_vis, 
                          (cx, cy), 
                          5, (0, 255, 0), -1)
                
                # 이미지 중심선
                image_center_x = self.warped_width // 2
                cv2.line(warped_vis,
                        (image_center_x, 0),
                        (image_center_x, self.warped_height),
                        (255, 255, 0), 2)
            
            # 원본 뷰로 역변환
            unwarped = self.perspective_transform.unwarp(warped_vis)
            
            # 원본 프레임에 오버레이
            roi = vis_frame[self.roi_top:self.roi_bottom, :]
            roi = cv2.addWeighted(roi, 0.6, unwarped, 0.4, 0)
            vis_frame[self.roi_top:self.roi_bottom, :] = roi
            
            # ROI 영역 표시
            cv2.rectangle(vis_frame, 
                         (0, self.roi_top), 
                         (self.image_width, self.roi_bottom),
                         (0, 255, 0), 2)
        else:
            # 기존 방식 (변환 없이)
            # ROI 영역 표시
            cv2.rectangle(vis_frame, 
                         (0, self.roi_top), 
                         (self.image_width, self.roi_bottom),
                         (0, 255, 0), 2)
            
            # 중앙선 그리기 (하얀색, 두껍게)
            if center_lane is not None:
                x1, y1, x2, y2 = center_lane[0]
                cv2.line(vis_frame, 
                        (x1, y1 + self.roi_top), 
                        (x2, y2 + self.roi_top),
                        (255, 255, 255), 4)  # 하얀색, 두껍게
            
            # 왼쪽 차선 그리기 (노란색)
            if left_lane is not None:
                x1, y1, x2, y2 = left_lane[0]
                cv2.line(vis_frame, 
                        (x1, y1 + self.roi_top), 
                        (x2, y2 + self.roi_top),
                        (0, 255, 255), 3)  # 노란색 (BGR)
            
            # 오른쪽 차선 그리기 (노란색)
            if right_lane is not None:
                x1, y1, x2, y2 = right_lane[0]
                cv2.line(vis_frame, 
                        (x1, y1 + self.roi_top), 
                        (x2, y2 + self.roi_top),
                        (0, 255, 255), 3)  # 노란색 (BGR)
            
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

