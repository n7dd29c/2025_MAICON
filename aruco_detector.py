# -*- coding: utf-8 -*-
"""
ArUco 마커 감지 모듈
마커 ID별 주행 명령 생성
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List


class ArUcoDetector:
    """ArUco 마커 감지 클래스"""
    
    def __init__(self, 
                 dictionary_id: int = 8,  # cv2.aruco.DICT_6X6_50
                 marker_size: float = 0.05):
        """
        Args:
            dictionary_id: ArUco 딕셔너리 ID (기본: 8 = DICT_6X6_50)
            marker_size: 마커 크기 (미터 단위, 카메라 캘리브레이션 시 사용)
        """
        # ArUco 딕셔너리 및 파라미터 초기화 (OpenCV 버전 호환)
        try:
            # OpenCV 4.7+ 버전
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            # OpenCV 4.6 이하 버전
            self.aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # 회전된 마커 감지를 위한 관대한 파라미터 설정
        self.aruco_params.minMarkerPerimeterRate = 0.01
        self.aruco_params.maxMarkerPerimeterRate = 20.0
        self.aruco_params.adaptiveThreshConstant = 3
        self.aruco_params.errorCorrectionRate = 1.0
        # 회전 각도 관련 파라미터
        self.aruco_params.polygonalApproxAccuracyRate = 0.05  # 다각형 근사 정확도 (더 관대하게)
        self.aruco_params.minCornerDistanceRate = 0.01  # 코너 간 최소 거리
        self.aruco_params.minDistanceToBorder = 1  # 경계로부터 최소 거리
        
        # 마커 ID별 동작 정의
        # 사용자가 마커를 인쇄/표시하여 사용
        self.marker_actions = {
            0: 'turn_right',    # ID 0 → 우회전
            1: 'turn_left',     # ID 1 → 좌회전
            2: 'go_straight',   # ID 2 → 직진
            3: 'stop',          # ID 3 → 정지
            # 추가 마커 ID는 여기에 정의
        }
        
        # 마커 감지 상태
        self.last_detected_marker = None
        self.marker_detection_count = 0
        self.min_detection_frames = 3  # 연속 3프레임 감지 시 명령 실행
        
    def detect_markers(self, frame: np.ndarray) -> Tuple[Optional[Dict], List]:
        """
        ArUco 마커 감지 (회전된 마커도 감지 가능)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (marker_command, all_markers): 마커 명령과 모든 감지된 마커 정보
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 전처리 방법들 (회전된 마커 감지 개선)
        preprocess_methods = [
            ("대비 향상", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)),
            ("원본", gray),
            ("이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("역이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
        ]
        
        all_markers = []
        marker_command = None
        corners = None
        ids = None
        
        # 여러 전처리 방법으로 시도 (회전된 마커 감지 개선)
        for method_name, processed_img in preprocess_methods:
            # ArUco 마커 감지
            corners, ids, rejected = cv2.aruco.detectMarkers(
                processed_img, self.aruco_dict, parameters=self.aruco_params
            )
            
            if ids is not None and len(ids) > 0:
                # 감지 성공 시 루프 종료
                break
        
        if ids is not None and len(ids) > 0:
            # 감지된 마커 정보 저장
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i][0]
                center = np.mean(marker_corners, axis=0).astype(int)
                
                all_markers.append({
                    'id': int(marker_id),
                    'corners': marker_corners,
                    'center': center,
                    'action': self.marker_actions.get(int(marker_id), 'go_straight')
                })
            
            # 첫 번째 마커의 동작 사용
            first_marker = all_markers[0]
            marker_id = first_marker['id']
            action = first_marker['action']
            
            # 연속 감지 확인 (오탐 방지)
            if self.last_detected_marker == marker_id:
                self.marker_detection_count += 1
            else:
                self.last_detected_marker = marker_id
                self.marker_detection_count = 1
            
            # 연속으로 충분히 감지되면 명령 생성
            if self.marker_detection_count >= self.min_detection_frames:
                marker_command = {
                    'action': action,
                    'marker_id': marker_id,
                    'center': first_marker['center'],
                    'corners': first_marker['corners']
                }
        else:
            # 마커 미감지 시 카운터 리셋
            self.last_detected_marker = None
            self.marker_detection_count = 0
        
        return marker_command, all_markers
    
    def draw_markers(self, frame: np.ndarray, markers: List[Dict]) -> np.ndarray:
        """
        ArUco 마커 시각화
        
        Args:
            frame: 입력 프레임
            markers: 감지된 마커 리스트
            
        Returns:
            vis_frame: 시각화된 프레임
        """
        vis_frame = frame.copy()
        
        for marker in markers:
            marker_id = marker['id']
            corners = marker['corners'].astype(int)
            center = marker['center']
            action = marker['action']
            
            # 마커 윤곽선 그리기
            cv2.aruco.drawDetectedMarkers(vis_frame, [corners.reshape(1, -1, 2)], 
                                         np.array([[marker_id]]))
            
            # 중심점 표시
            cv2.circle(vis_frame, tuple(center), 5, (0, 255, 0), -1)
            
            # 마커 ID 및 동작 표시
            text = f"ID:{marker_id} ({action})"
            cv2.putText(vis_frame, text,
                       (int(center[0]) - 50, int(center[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame
    
    def set_marker_action(self, marker_id: int, action: str):
        """
        마커 ID별 동작 설정
        
        Args:
            marker_id: 마커 ID
            action: 동작 ('turn_right', 'turn_left', 'go_straight', 'stop')
        """
        self.marker_actions[marker_id] = action

