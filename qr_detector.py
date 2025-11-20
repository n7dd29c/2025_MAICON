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
        self.min_detection_frames = 2  # 연속 2프레임 감지 시 인식
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        QR 코드 감지
        
        Args:
            frame: 입력 프레임
            
        Returns:
            qr_codes: 감지된 QR 코드 리스트 [{'data': str, 'points': np.ndarray, 'center': tuple}]
        """
        if self.qr_detector is None:
            return []
        
        qr_codes = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # QR 코드 감지
            retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(gray)
            
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
        except Exception as e:
            # OpenCV 버전에 따라 다른 메서드 시도
            try:
                retval, decoded_info, points = self.qr_detector.detectAndDecodeMulti(gray)
                if retval and decoded_info is not None:
                    for i, data in enumerate(decoded_info):
                        if data:
                            qr_points = points[i] if points is not None else None
                            if qr_points is not None and len(qr_points) > 0:
                                center = np.mean(qr_points.reshape(-1, 2), axis=0).astype(int)
                            else:
                                center = (0, 0)
                            
                            qr_codes.append({
                                'data': data,
                                'points': qr_points,
                                'center': tuple(center)
                            })
            except Exception as e2:
                # 단일 QR 코드 감지 시도
                try:
                    retval, decoded_info, points = self.qr_detector.detectAndDecode(gray)
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
                except:
                    pass
        
        return qr_codes
    
    def draw_codes(self, frame: np.ndarray, qr_codes: List[Dict]) -> np.ndarray:
        """
        QR 코드 시각화
        
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
            
            # QR 코드 윤곽선 그리기
            if points is not None:
                points = points.astype(int)
                if len(points.shape) == 2:
                    # 4개의 점을 연결
                    for i in range(len(points)):
                        pt1 = tuple(points[i])
                        pt2 = tuple(points[(i + 1) % len(points)])
                        cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)
            
            # 중심점 표시
            cv2.circle(vis_frame, center, 5, (0, 255, 0), -1)
            
            # QR 코드 데이터 표시 (처음 20자만)
            display_data = data[:20] + "..." if len(data) > 20 else data
            text = f"QR: {display_data}"
            cv2.putText(vis_frame, text,
                       (center[0] - 50, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame

