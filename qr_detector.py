# -*- coding: utf-8 -*-
"""
QR 코드 감지 모듈
QR 코드 인식 및 LED 제어
"""
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple


class QRDetector:
    """QR 코드 감지 클래스"""
    
    def __init__(self):
        """QR 코드 감지기 초기화"""
        # OpenCV QRCodeDetector 초기화
        try:
            self.qr_detector = cv2.QRCodeDetector()
        except AttributeError:
            # OpenCV 버전에 따라 다를 수 있음
            try:
                self.qr_detector = cv2.wechat_qrcode.QRCodeDetector()
            except:
                print("경고: QR 코드 감지기를 초기화할 수 없습니다.")
                self.qr_detector = None
        
        # 감지 상태
        self.last_detected_qr = None
        self.qr_detection_count = 0
        self.min_detection_frames = 1  # 연속 1프레임 감지 시 인식 (완화)
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        QR 코드 감지 (강화된 전처리)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            qr_codes: 감지된 QR 코드 리스트 [{'data': str, 'points': np.ndarray, 'center': tuple}]
        """
        if self.qr_detector is None:
            return []
        
        if frame is None:
            return []
        
        qr_codes = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            return []
        
        # 여러 전처리 방법 시도
        preprocess_methods = [
            ("원본", gray),
            ("대비 향상", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)),
            ("가우시안 블러", cv2.GaussianBlur(gray, (5, 5), 0)),
            ("샤프닝", cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))),
            ("이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("역이진화", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
        ]
        
        for method_name, processed_img in preprocess_methods:
            try:
                # QR 코드 감지 (여러 메서드 시도)
                # 방법 1: detectAndDecodeMulti
                try:
                    retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(processed_img)
                    
                    if retval and decoded_info is not None and len(decoded_info) > 0:
                        # 여러 QR 코드 감지 가능
                        for i, data in enumerate(decoded_info):
                            if data:  # 데이터가 있는 경우만
                                qr_points = points[i] if points is not None else None
                                
                                # 중심점 계산
                                if qr_points is not None and len(qr_points) > 0:
                                    center = np.mean(qr_points.reshape(-1, 2), axis=0).astype(int)
                                else:
                                    center = (0, 0)
                                
                                qr_codes.append({
                                    'data': data,
                                    'points': qr_points,
                                    'center': tuple(center)
                                })
                        
                        if qr_codes:
                            print(f"QR 코드 감지 성공: {method_name}, 데이터: {[qr['data'] for qr in qr_codes]}")
                            break
                except:
                    pass
                
                # 방법 2: detectAndDecode (단일)
                try:
                    retval, decoded_info, points = self.qr_detector.detectAndDecode(processed_img)
                    if retval and decoded_info:
                        if points is not None and len(points) > 0:
                            center = np.mean(points.reshape(-1, 2), axis=0).astype(int)
                        else:
                            center = (0, 0)
                        
                        qr_codes.append({
                            'data': decoded_info,
                            'points': points,
                            'center': tuple(center)
                        })
                        
                        if qr_codes:
                            print(f"QR 코드 감지 성공: {method_name}, 데이터: {decoded_info}")
                            break
                except:
                    pass
                    
            except Exception as e:
                continue
        
        # 회전된 QR 코드 감지 시도
        if not qr_codes:
            h, w = gray.shape
            center = (w // 2, h // 2)
            rotation_angles = [90, 180, 270]
            
            for angle in rotation_angles:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, M, (w, h))
                
                try:
                    retval, decoded_info, points = self.qr_detector.detectAndDecode(rotated)
                    if retval and decoded_info:
                        if points is not None and len(points) > 0:
                            # 회전된 좌표를 원본 좌표로 변환
                            points_rotated = points.reshape(-1, 2)
                            # 역회전 변환
                            M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                            points_original = cv2.transform(points_rotated.reshape(1, -1, 2), M_inv).reshape(-1, 2)
                            center = np.mean(points_original, axis=0).astype(int)
                        else:
                            center = (0, 0)
                        
                        qr_codes.append({
                            'data': decoded_info,
                            'points': points_original if 'points_original' in locals() else points,
                            'center': tuple(center)
                        })
                        
                        if qr_codes:
                            print(f"QR 코드 감지 성공: {angle}도 회전, 데이터: {decoded_info}")
                            break
                except:
                    continue
        
        # QR 코드가 감지되지 않았을 때 주기적으로 로그 출력 (디버깅용)
        if not qr_codes:
            self.qr_detection_count = 0
        else:
            self.qr_detection_count += 1
        
        return qr_codes
    
    def draw_shape(self, frame: np.ndarray, shape_char: str, color: Tuple[int, int, int], 
                   center: Tuple[int, int], size: int = 30) -> np.ndarray:
        """
        모양 그리기 (베이스라인 코드 방식)
        
        Args:
            frame: 입력 프레임
            shape_char: 모양 문자 ('O', 'X', '#', 'R')
            color: 색상 (B, G, R)
            center: 중심점 (x, y)
            size: 크기
            
        Returns:
            vis_frame: 시각화된 프레임
        """
        vis_frame = frame.copy()
        cx, cy = center
        
        if shape_char == 'O':
            # 원형
            cv2.circle(vis_frame, center, size, color, 3)
        elif shape_char == 'X':
            # X 모양
            offset = int(size * 0.7)
            cv2.line(vis_frame, (cx - offset, cy - offset), (cx + offset, cy + offset), color, 3)
            cv2.line(vis_frame, (cx - offset, cy + offset), (cx + offset, cy - offset), color, 3)
        elif shape_char == '#':
            # # 모양 (격자)
            offset = int(size * 0.7)
            # 수직선 2개
            cv2.line(vis_frame, (cx - offset, cy - size), (cx - offset, cy + size), color, 3)
            cv2.line(vis_frame, (cx + offset, cy - size), (cx + offset, cy + size), color, 3)
            # 수평선 2개
            cv2.line(vis_frame, (cx - size, cy - offset), (cx + size, cy - offset), color, 3)
            cv2.line(vis_frame, (cx - size, cy + offset), (cx + size, cy + offset), color, 3)
        elif shape_char == 'R':
            # 사각형 (Rectangle)
            offset = int(size * 0.7)
            cv2.rectangle(vis_frame, (cx - offset, cy - offset), (cx + offset, cy + offset), color, 3)
        else:
            # 기본: 사각형
            offset = int(size * 0.7)
            cv2.rectangle(vis_frame, (cx - offset, cy - offset), (cx + offset, cy + offset), color, 3)
        
        return vis_frame
    
    def parse_qr_data(self, data: str) -> Tuple[str, Tuple[int, int, int]]:
        """
        QR 코드 데이터 파싱 (ID_SHAPE_COLOR 형식)
        
        Args:
            data: QR 코드 데이터 (예: "ID_O_R", "ID_X_G", "ID_#_B")
            
        Returns:
            (shape_char, color): 모양 문자와 색상 (RGB 형식, OpenCV BGR로 변환)
        """
        # 기본값
        shape_char = 'R'  # 사각형
        color = (0, 50, 0)  # 녹색 (RGB 형식)
        
        # ID_SHAPE_COLOR 형식 파싱
        if data.startswith("ID_"):
            parts = data.split("_")
            if len(parts) >= 3:
                shape_str = parts[1]  # O, X, #
                color_str = parts[2]  # R, G, B
                
                # 모양 결정
                if shape_str == 'O':
                    shape_char = 'O'
                elif shape_str == 'X':
                    shape_char = 'X'
                elif shape_str == '#' or shape_str == '#':
                    shape_char = '#'
                else:
                    shape_char = 'R'  # 기본: 사각형
                
                # 색상 결정 (사용자 제공 형식: (50,0,0) = 빨강, (0,50,0) = 녹색, (0,0,50) = 파랑)
                # OpenCV는 BGR을 사용하므로 변환 필요
                if color_str.upper() == 'R':
                    color = (0, 0, 50)  # 빨간색 (BGR): RGB(50,0,0) → BGR(0,0,50)
                elif color_str.upper() == 'G':
                    color = (0, 50, 0)  # 녹색 (BGR): RGB(0,50,0) → BGR(0,50,0)
                elif color_str.upper() == 'B':
                    color = (50, 0, 0)  # 파란색 (BGR): RGB(0,0,50) → BGR(50,0,0)
        
        return shape_char, color
    
    def draw_codes(self, frame: np.ndarray, qr_codes: List[Dict]) -> np.ndarray:
        """
        QR 코드 시각화 (베이스라인 코드 방식: ID_SHAPE_COLOR 형식)
        
        Args:
            frame: 입력 프레임
            qr_codes: 감지된 QR 코드 리스트
            
        Returns:
            vis_frame: 시각화된 프레임
        """
        vis_frame = frame.copy()
        
        for qr in qr_codes:
            data = qr['data']
            points = qr['points']
            center = qr['center']
            
            # QR 코드 데이터 파싱 (ID_SHAPE_COLOR 형식)
            shape_char, color = self.parse_qr_data(data)
            
            # 바운딩 박스 크기 계산 (모양 크기 결정용)
            if points is not None and len(points) > 0:
                points = points.astype(int)
                if len(points.shape) == 2:
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    width = int((np.max(x_coords) - np.min(x_coords)) * 0.5)
                    height = int((np.max(y_coords) - np.min(y_coords)) * 0.5)
                    size = max(width, height, 20)  # 최소 20픽셀
                else:
                    size = 30
            else:
                size = 30
            
            # 모양 그리기
            vis_frame = self.draw_shape(vis_frame, shape_char, color, center, size)
            
            # QR 코드 데이터 표시
            display_data = data[:20] + "..." if len(data) > 20 else data
            text = f"QR: {display_data}"
            cv2.putText(vis_frame, text,
                       (center[0] - 50, center[1] - size - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame

