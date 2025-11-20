# -*- coding: utf-8 -*-
"""
ArUco 마커 감지 모듈
베이스라인 코드 기준 - 간단하고 효과적
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List


class ArUcoDetector:
    """ArUco 마커 감지 클래스 (베이스라인 코드 기준)"""
    
    def __init__(self, dictionary_id: int = 10):  # cv2.aruco.DICT_4X4_50
        """
        Args:
            dictionary_id: ArUco 딕셔너리 ID (기본: 10 = DICT_4X4_50)
        """
        # 베이스라인 코드처럼 딕셔너리 초기화
        try:
            # OpenCV 4.7+ 버전
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            # OpenCV 4.6 이하 버전
            self.aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # 베이스라인 코드처럼 ArucoDetector 사용 시도
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except:
            self.detector = None
        
        # 마커 ID별 동작 정의 (config.py에서 설정 가능)
        try:
            from config import ARUCO_MARKER_ACTIONS
            self.marker_actions = ARUCO_MARKER_ACTIONS.copy()
        except ImportError:
            # config.py에 없으면 기본값 사용
            self.marker_actions = {
                0: 'turn_right',    # ID 0 → 우회전
                1: 'turn_left',     # ID 1 → 좌회전
                2: 'go_straight',   # ID 2 → 직진
                3: 'stop',          # ID 3 → 정지
            }
    
    def detect_markers(self, frame: np.ndarray) -> Tuple[Optional[Dict], List]:
        """
        ArUco 마커 감지 (베이스라인 코드와 동일)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (marker_command, all_markers): 마커 명령과 모든 감지된 마커 정보
        """
        all_markers = []
        marker_command = None
        
        # 베이스라인 코드처럼 간단하게 감지
        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
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
            
            marker_command = {
                'action': action,
                'marker_id': marker_id,
                'center': first_marker['center'],
                'corners': first_marker['corners']
            }
        
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
        
        if not markers:
            return vis_frame
        
        # 마커 정보 수집
        marker_corners_list = []
        marker_ids_list = []
        
        for marker in markers:
            marker_id = marker['id']
            corners = marker['corners'].astype(int)
            center = marker['center']
            action = marker['action']
            
            marker_corners_list.append(corners.reshape(1, -1, 2))
            marker_ids_list.append(marker_id)
            
            # 중심점 표시
            cv2.circle(vis_frame, tuple(center), 5, (0, 255, 0), -1)
            
            # 마커 ID 및 동작 표시
            text = f"ID:{marker_id} ({action})"
            cv2.putText(vis_frame, text,
                       (int(center[0]) - 50, int(center[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 베이스라인 코드처럼 drawDetectedMarkers 사용
        if marker_corners_list:
            corners_array = np.concatenate(marker_corners_list, axis=0)
            ids_array = np.array(marker_ids_list).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(vis_frame, corners_array, ids_array)
        
        return vis_frame
    
    def set_marker_action(self, marker_id: int, action: str):
        """
        마커 ID별 동작 설정
        
        Args:
            marker_id: 마커 ID
            action: 동작 ('turn_right', 'turn_left', 'go_straight', 'stop')
        """
        self.marker_actions[marker_id] = action
