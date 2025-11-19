"""
객체 디텍션 모듈
포트홀 감지 및 장애물 인식
Bird's Eye View 적용
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from config import *
from perspective_transform import PerspectiveTransform


class ObjectDetector:
    """객체 디텍터 클래스 - 포트홀 감지"""
    
    def __init__(self,
                 image_width: int = 640,
                 image_height: int = 480,
                 roi_ratio: float = 0.6,
                 use_bird_view: bool = True):
        """
        Args:
            image_width: 이미지 너비
            image_height: 이미지 높이
            roi_ratio: 관심 영역(ROI) 비율 (하단 부분만 사용)
            use_bird_view: Bird's Eye View 사용 여부
        """
        self.image_width = image_width
        self.image_height = image_height
        self.roi_ratio = roi_ratio
        self.use_bird_view = use_bird_view
        
        # ROI 영역 계산
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
        
        # 포트홀 감지 설정
        # 포트홀은 일반적으로 어두운 색상 (검은색, 회색)
        self.pothole_lower = np.array([0, 0, 0])  # HSV - 어두운 영역
        self.pothole_upper = np.array([180, 255, 80])  # HSV
        
        # 최소/최대 포트홀 크기 (픽셀)
        self.min_pothole_area = 100  # 최소 면적
        self.max_pothole_area = 50000  # 최대 면적
        
        # 포트홀 감지 임계값
        self.depth_threshold = 30  # 깊이 차이 임계값
        
    def detect_potholes(self, frame: np.ndarray) -> List[Dict]:
        """
        포트홀 감지
        
        Args:
            frame: 입력 프레임
            
        Returns:
            potholes: 포트홀 정보 리스트 [{'bbox': (x, y, w, h), 'center': (cx, cy), 'area': area}]
        """
        # Bird's Eye View 변환
        if self.use_bird_view and self.perspective_transform:
            warped = self.perspective_transform.warp(frame)
            roi = warped
        else:
            # ROI 추출
            roi = frame[self.roi_top:self.roi_bottom, :]
        
        # HSV 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 어두운 영역 마스크 (포트홀 후보)
        dark_mask = cv2.inRange(hsv, self.pothole_lower, self.pothole_upper)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 적응형 임계값 처리 (포트홀은 주변보다 어둡게 보임)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 마스크 결합
        combined_mask = cv2.bitwise_and(dark_mask, adaptive_thresh)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        potholes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 크기 필터링
            if area < self.min_pothole_area or area > self.max_pothole_area:
                continue
            
            # 바운딩 박스 계산
            x, y, w, h = cv2.boundingRect(contour)
            
            # 종횡비 필터링 (포트홀은 대체로 원형 또는 타원형)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # 중심점 계산
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            # 좌표 변환
            if self.use_bird_view and self.perspective_transform:
                # Bird's Eye View에서는 변환된 이미지 좌표 사용
                # 원본 이미지 좌표로 역변환
                orig_center = self.perspective_transform.transform_point((cx, cy), inverse=True)
                orig_bbox = (
                    self.perspective_transform.transform_point((x, y), inverse=True)[0],
                    self.perspective_transform.transform_point((x, y), inverse=True)[1] - self.roi_top,
                    w, h
                )
                global_y = orig_center[1]
            else:
                # ROI 좌표를 전체 이미지 좌표로 변환
                global_y = cy + self.roi_top
                orig_center = (cx, global_y)
                orig_bbox = (x, y, w, h)
            
            potholes.append({
                'bbox': orig_bbox,
                'center': orig_center,
                'area': area,
                'roi_center': (cx, cy),  # ROI 내부 좌표
                'warped_center': (cx, cy) if self.use_bird_view else None  # 변환된 이미지 좌표
            })
        
        # 면적 기준으로 정렬 (큰 것부터)
        potholes.sort(key=lambda x: x['area'], reverse=True)
        
        return potholes
    
    def is_pothole_in_path(self, 
                          pothole: Dict,
                          left_lane: Optional[np.ndarray],
                          right_lane: Optional[np.ndarray],
                          image_width: int = 640) -> bool:
        """
        포트홀이 주행 경로에 있는지 확인
        
        Args:
            pothole: 포트홀 정보
            left_lane: 왼쪽 차선
            right_lane: 오른쪽 차선
            image_width: 이미지 너비
            
        Returns:
            is_in_path: 주행 경로에 있는지 여부
        """
        if left_lane is None or right_lane is None:
            # 차선이 없으면 중심선 기준으로 판단
            center_x = image_width // 2
            pothole_x = pothole['center'][0]
            return abs(pothole_x - center_x) < image_width * 0.2
        
        # 차선 내부 영역 계산
        pothole_x, pothole_y = pothole['center']
        
        # 하단 부분의 차선 위치 추정
        # 왼쪽 차선의 하단 x 좌표
        left_x = left_lane[0][0] if left_lane is not None else 0
        # 오른쪽 차선의 하단 x 좌표
        right_x = right_lane[0][0] if right_lane is not None else image_width
        
        # 포트홀이 차선 내부에 있는지 확인
        # 약간의 여유를 두고 판단 (포트홀 크기 고려)
        margin = pothole['bbox'][2] // 2  # 포트홀 너비의 절반
        
        is_in_path = (left_x + margin < pothole_x < right_x - margin)
        
        return is_in_path
    
    def draw_potholes(self, 
                     frame: np.ndarray,
                     potholes: List[Dict],
                     in_path_only: bool = False,
                     left_lane: Optional[np.ndarray] = None,
                     right_lane: Optional[np.ndarray] = None) -> np.ndarray:
        """
        포트홀 시각화
        
        Args:
            frame: 입력 프레임
            potholes: 포트홀 리스트
            in_path_only: 주행 경로에 있는 포트홀만 표시
            left_lane: 왼쪽 차선
            right_lane: 오른쪽 차선
        """
        vis_frame = frame.copy()
        
        if self.use_bird_view and self.perspective_transform:
            # Bird's Eye View에서 포트홀 그리기
            warped_vis = self.perspective_transform.warp(vis_frame)
            
            for pothole in potholes:
                # 주행 경로에 있는지 확인
                if in_path_only:
                    if not self.is_pothole_in_path(
                        pothole, left_lane, right_lane, self.image_width
                    ):
                        continue
                
                # 변환된 이미지 좌표 사용
                if pothole.get('warped_center') is not None:
                    cx, cy = pothole['warped_center']
                    x, y, w, h = pothole['bbox']
                    # bbox도 변환된 좌표로 변환 필요
                    # 간단히 warped_center 기준으로 그리기
                    bbox_size = int(np.sqrt(pothole['area']) * 0.8)
                    x = cx - bbox_size // 2
                    y = cy - bbox_size // 2
                    w = h = bbox_size
                else:
                    # 변환된 좌표가 없으면 원본 좌표를 변환
                    orig_cx, orig_cy = pothole['center']
                    # 원본 좌표를 변환된 좌표로 변환
                    warped_point = self.perspective_transform.transform_point((orig_cx, orig_cy), inverse=False)
                    cx, cy = warped_point
                    bbox_size = int(np.sqrt(pothole['area']) * 0.8)
                    x = cx - bbox_size // 2
                    y = cy - bbox_size // 2
                    w = h = bbox_size
                
                # 주행 경로에 있는 포트홀은 빨간색, 아닌 것은 노란색
                if in_path_only and self.is_pothole_in_path(
                    pothole, left_lane, right_lane, self.image_width
                ):
                    color = (0, 0, 255)  # 빨간색
                    thickness = 3
                else:
                    color = (0, 255, 255)  # 노란색
                    thickness = 2
                
                # 바운딩 박스 그리기
                cv2.rectangle(warped_vis,
                             (x, y),
                             (x + w, y + h),
                             color, thickness)
                
                # 중심점 표시
                cv2.circle(warped_vis, (cx, cy), 5, color, -1)
                
                # 텍스트 표시
                cv2.putText(warped_vis, f"Pothole: {pothole['area']:.0f}",
                           (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 원본 뷰로 역변환
            unwarped = self.perspective_transform.unwarp(warped_vis)
            
            # 원본 프레임에 오버레이
            roi = vis_frame[self.roi_top:self.roi_bottom, :]
            roi = cv2.addWeighted(roi, 0.6, unwarped, 0.4, 0)
            vis_frame[self.roi_top:self.roi_bottom, :] = roi
        else:
            # 기존 방식 (변환 없이)
            for pothole in potholes:
                # 주행 경로에 있는지 확인
                if in_path_only:
                    if not self.is_pothole_in_path(
                        pothole, left_lane, right_lane, self.image_width
                    ):
                        continue
                
                x, y, w, h = pothole['bbox']
                cx, cy = pothole['center']
                
                # ROI 좌표를 전체 이미지 좌표로 변환
                global_y = y + self.roi_top
                
                # 주행 경로에 있는 포트홀은 빨간색, 아닌 것은 노란색
                if in_path_only and self.is_pothole_in_path(
                    pothole, left_lane, right_lane, self.image_width
                ):
                    color = (0, 0, 255)  # 빨간색
                    thickness = 3
                else:
                    color = (0, 255, 255)  # 노란색
                    thickness = 2
                
                # 바운딩 박스 그리기
                cv2.rectangle(vis_frame,
                             (x, global_y),
                             (x + w, global_y + h),
                             color, thickness)
                
                # 중심점 표시
                cv2.circle(vis_frame, (cx, cy), 5, color, -1)
                
                # 텍스트 표시
                cv2.putText(vis_frame, f"Pothole: {pothole['area']:.0f}",
                           (x, global_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame

