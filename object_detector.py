# -*- coding: utf-8 -*-
# 객체 디텍션 모듈
# 포트홀 감지 및 장애물 인식
# Bird's Eye View 적용
import os
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from config import *
from perspective_transform import PerspectiveTransform

# YOLO 관련 설정 명시적 import (호환성)
try:
    from config import YOLO_MODEL_PATH
except ImportError:
    # config.py에 없으면 기본값 사용
    YOLO_MODEL_PATH = "yolo/Object7.v7-maicon_mortar_background.yolov8/yolov8n/weights/best.onnx"

# YOLO ONNX Runtime 추론 (선택적)
YOLO_AVAILABLE = False
try:
    import onnxruntime as ort
    from yolo_inference import YOLOONNX
    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"경고: YOLO 추론 모듈을 사용할 수 없습니다. 전통적인 CV 방법을 사용합니다.")
    print(f"  오류: {e}")
    print(f"  설치: pip install onnxruntime")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"경고: YOLO 추론 모듈 로드 실패: {e}")
    print(f"  전통적인 CV 방법을 사용합니다.")


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
        
        # YOLO 초기화 (설정에서 활성화된 경우)
        self.use_yolo = USE_YOLO and YOLO_AVAILABLE
        self.yolo_model = None
        
        # 클래스 설정
        self.class_names = YOLO_CLASS_NAMES
        self.avoidance_classes = YOLO_AVOIDANCE_CLASSES
        self.display_only_classes = YOLO_DISPLAY_ONLY_CLASSES
        
        # 클래스 ID 매핑 (이름 -> ID)
        self.class_id_map = {name: idx for idx, name in enumerate(self.class_names)}
        # 회피가 필요한 클래스 ID
        self.avoidance_class_ids = [self.class_id_map.get(name, -1) 
                                    for name in self.avoidance_classes 
                                    if name in self.class_id_map]
        
        if self.use_yolo:
            try:
                # YOLO_MODEL_PATH 가져오기 (명시적 import)
                try:
                    from config import YOLO_MODEL_PATH
                except ImportError:
                    # config.py에 없으면 기본값 사용
                    YOLO_MODEL_PATH = "yolo/Object7.v7-maicon_mortar_background.yolov8/yolov8n/weights/best.onnx"
                
                model_path = YOLO_MODEL_PATH
                
                # 여러 경로 시도
                possible_paths = [
                    model_path,  # 원본 경로
                    os.path.join(os.path.dirname(__file__), model_path),  # 현재 디렉토리 기준
                    os.path.join(os.path.dirname(__file__), '..', model_path),  # 상위 디렉토리
                    os.path.abspath(model_path),  # 절대 경로
                    os.path.join(os.getcwd(), model_path),  # 작업 디렉토리 기준
                ]
                
                found_path = None
                for path in possible_paths:
                    path = os.path.normpath(path)
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if found_path:
                    try:
                        self.yolo_model = YOLOONNX(
                            onnx_path=found_path,
                            conf_threshold=YOLO_CONF_THRESHOLD,
                            iou_threshold=YOLO_IOU_THRESHOLD
                        )
                        print(f"✓ YOLO 모델 로드 완료: {found_path}")
                        print(f"  감지 가능한 클래스: {self.class_names}")
                        print(f"  회피 대상 클래스: {self.avoidance_classes}")
                    except Exception as e:
                        print(f"✗ YOLO 모델 초기화 실패: {e}")
                        import traceback
                        traceback.print_exc()
                        print("전통적인 CV 방법을 사용합니다.")
                        self.use_yolo = False
                        self.yolo_model = None
                else:
                    print(f"경고: YOLO ONNX 파일을 찾을 수 없습니다.")
                    print(f"  시도한 경로:")
                    for path in possible_paths:
                        print(f"    - {os.path.normpath(path)}")
                    print("전통적인 CV 방법을 사용합니다.")
                    self.use_yolo = False
                    self.yolo_model = None
            except Exception as e:
                print(f"YOLO 초기화 실패: {e}")
                import traceback
                traceback.print_exc()
                print("전통적인 CV 방법을 사용합니다.")
                self.use_yolo = False
                self.yolo_model = None
        
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        객체 감지 (YOLO 또는 전통적인 CV 방법)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (avoidance_objects, all_objects): 
                - avoidance_objects: 회피가 필요한 객체 리스트 (포트홀 등)
                - all_objects: 모든 감지된 객체 리스트 (표시용)
        """
        # YOLO 사용 시
        if self.use_yolo and self.yolo_model is not None:
            try:
                # YOLO 추론 (전체 프레임 사용)
                detections = self.yolo_model.predict(frame)
                
                # 모든 객체와 회피 대상 객체 분리
                all_objects = []
                avoidance_objects = []
                
                for det in detections:
                    class_id = det.get('class', -1)
                    class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # 객체 정보 구성
                    obj_info = {
                        'bbox': det['bbox'],
                        'center': det['center'],
                        'area': det['area'],
                        'conf': det['conf'],
                        'class': class_id,
                        'class_name': class_name
                    }
                    
                    all_objects.append(obj_info)
                    
                    # 회피 대상인지 확인
                    if class_id in self.avoidance_class_ids:
                        # 크기 필터링 (포트홀 등 회피 대상만)
                        if self.min_pothole_area <= obj_info['area'] <= self.max_pothole_area:
                            avoidance_objects.append(obj_info)
                
                return avoidance_objects, all_objects
            except Exception as e:
                print(f"YOLO 추론 오류: {e}")
                # 오류 시 전통적인 방법으로 폴백
                pass
        
        # 전통적인 CV 방법 (포트홀만 감지)
        potholes = self.detect_potholes_cv(frame)
        return potholes, potholes
    
    def detect_potholes(self, frame: np.ndarray) -> List[Dict]:
        """
        포트홀 감지 (하위 호환성을 위한 메서드)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            potholes: 포트홀 정보 리스트 [{'bbox': (x, y, w, h), 'center': (cx, cy), 'area': area, 'conf': conf}]
        """
        avoidance_objects, _ = self.detect_objects(frame)
        return avoidance_objects
    
    def detect_potholes_cv(self, frame: np.ndarray) -> List[Dict]:
        """
        전통적인 CV 방법으로 포트홀 감지
        
        Args:
            frame: 입력 프레임
            
        Returns:
            potholes: 포트홀 정보 리스트 [{'bbox': (x, y, w, h), 'center': (cx, cy), 'area': area, 'conf': conf, 'class': class_id, 'class_name': class_name}]
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
                'conf': 0.8,  # CV 방법은 신뢰도 없으므로 기본값
                'class': 0,  # 포트홀 클래스 ID
                'class_name': 'pothole',
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
    
    def draw_objects(self,
                     frame: np.ndarray,
                     objects: List[Dict],
                     show_class_name: bool = True) -> np.ndarray:
        """
        모든 객체 시각화
        
        Args:
            frame: 입력 프레임
            objects: 객체 리스트
            show_class_name: 클래스 이름 표시 여부
        
        Returns:
            vis_frame: 시각화된 프레임
        """
        vis_frame = frame.copy()
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            cx, cy = obj['center']
            conf = obj.get('conf', 0.0)
            class_name = obj.get('class_name', 'unknown')
            class_id = obj.get('class', -1)
            
            # 회피 대상인지 확인
            is_avoidance = class_id in self.avoidance_class_ids if hasattr(self, 'avoidance_class_ids') else False
            
            # 색상 설정
            if is_avoidance:
                color = (0, 0, 255)  # 빨간색 (회피 대상)
                thickness = 3
            else:
                color = (0, 255, 255)  # 노란색 (표시만)
                thickness = 2
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
            
            # 클래스 이름과 신뢰도 표시
            if show_class_name:
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y - 10, label_size[1])
                cv2.rectangle(vis_frame, (x, label_y - label_size[1] - 5), 
                             (x + label_size[0], label_y + 5), color, -1)
                cv2.putText(vis_frame, label, (x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame

