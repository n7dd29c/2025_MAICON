"""
Perspective Transform 모듈
Bird's Eye View 변환
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from config import (
    BIRD_VIEW_TOP_WIDTH_RATIO,
    BIRD_VIEW_BOTTOM_MARGIN_RATIO,
    BIRD_VIEW_TOP_OFFSET_RATIO
)


class PerspectiveTransform:
    """Bird's Eye View 변환 클래스"""
    
    def __init__(self,
                 image_width: int = 640,
                 image_height: int = 480,
                 roi_ratio: float = 0.6):
        """
        Args:
            image_width: 원본 이미지 너비
            image_height: 원본 이미지 높이
            roi_ratio: ROI 비율 (하단 부분만 사용)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.roi_ratio = roi_ratio
        
        # ROI 영역 계산
        self.roi_top = int(image_height * (1 - roi_ratio))
        self.roi_bottom = image_height
        
        # 변환 매트릭스 초기화
        self.M: Optional[np.ndarray] = None
        self.M_inv: Optional[np.ndarray] = None
        self.warped_width: int = image_width
        self.warped_height: int = int(image_height * roi_ratio)
        
        # 변환 매트릭스 계산
        self._calculate_transform_matrix()
    
    def _calculate_transform_matrix(self):
        """Perspective transform 매트릭스 계산"""
        # 원본 이미지의 ROI 영역에서 사다리꼴 영역 정의
        # 카메라가 아래를 향하므로 상단도 더 넓게 설정
        
        # 원본 포인트 (사다리꼴)
        # 하단: 넓게 (차선이 넓게 보임)
        bottom_width = self.image_width
        bottom_margin = int(self.image_width * BIRD_VIEW_BOTTOM_MARGIN_RATIO)
        
        # 상단: config에서 설정한 비율 사용 (카메라가 아래를 향하므로 더 넓게)
        top_width = int(self.image_width * BIRD_VIEW_TOP_WIDTH_RATIO)
        top_margin = int((self.image_width - top_width) / 2)
        
        # ROI 높이 계산
        roi_height = self.roi_bottom - self.roi_top
        
        # 상단 포인트를 약간 아래로 조정 (카메라가 아래를 향하므로)
        top_y_offset = int(roi_height * BIRD_VIEW_TOP_OFFSET_RATIO)
        
        src_points = np.float32([
            [bottom_margin, roi_height],  # 하단 왼쪽
            [self.image_width - bottom_margin, roi_height],  # 하단 오른쪽
            [top_margin, top_y_offset],  # 상단 왼쪽 (약간 아래로)
            [self.image_width - top_margin, top_y_offset]  # 상단 오른쪽 (약간 아래로)
        ])
        
        # 변환 후 포인트 (직사각형)
        dst_points = np.float32([
            [0, self.warped_height],  # 하단 왼쪽
            [self.warped_width, self.warped_height],  # 하단 오른쪽
            [0, 0],  # 상단 왼쪽
            [self.warped_width, 0]  # 상단 오른쪽
        ])
        
        # 변환 매트릭스 계산
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    
    def warp(self, image: np.ndarray) -> np.ndarray:
        """
        원본 이미지를 Bird's Eye View로 변환
        
        Args:
            image: 원본 이미지
            
        Returns:
            warped: 변환된 이미지
        """
        # ROI 추출
        roi = image[self.roi_top:self.roi_bottom, :]
        
        # Perspective transform 적용
        warped = cv2.warpPerspective(
            roi, self.M,
            (self.warped_width, self.warped_height),
            flags=cv2.INTER_LINEAR
        )
        
        return warped
    
    def unwarp(self, warped_image: np.ndarray) -> np.ndarray:
        """
        Bird's Eye View를 원본 뷰로 역변환
        
        Args:
            warped_image: 변환된 이미지
            
        Returns:
            unwarped: 역변환된 이미지
        """
        if self.M_inv is None:
            return warped_image
        
        unwarped = cv2.warpPerspective(
            warped_image, self.M_inv,
            (self.image_width, self.roi_bottom - self.roi_top),
            flags=cv2.INTER_LINEAR
        )
        
        return unwarped
    
    def transform_point(self, point: Tuple[int, int], inverse: bool = False) -> Tuple[int, int]:
        """
        점 좌표 변환
        
        Args:
            point: (x, y) 좌표
            inverse: True면 역변환
            
        Returns:
            transformed_point: 변환된 좌표
        """
        if inverse:
            if self.M_inv is None:
                return point
            M = self.M_inv
        else:
            if self.M is None:
                return point
            M = self.M
        
        # ROI 좌표로 변환 (원본 이미지 좌표 → ROI 좌표)
        if not inverse:
            x, y = point
            y_roi = y - self.roi_top
            point_array = np.array([[[x, y_roi]]], dtype=np.float32)
        else:
            point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # 변환 적용
        transformed = cv2.perspectiveTransform(point_array, M)
        x, y = int(transformed[0][0][0]), int(transformed[0][0][1])
        
        # 역변환 시 ROI 좌표를 원본 이미지 좌표로 변환
        if inverse:
            y += self.roi_top
        
        return (x, y)
    
    def get_warped_size(self) -> Tuple[int, int]:
        """변환된 이미지 크기 반환"""
        return (self.warped_width, self.warped_height)

