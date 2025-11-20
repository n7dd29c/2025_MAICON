# -*- coding: utf-8 -*-
# YOLO ONNX Runtime 추론 모듈
# Jetson Nano 최적화된 YOLO 추론 (Python 3.8 호환)
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("경고: ONNX Runtime을 사용할 수 없습니다.")
    print("  설치: pip install onnxruntime-gpu (Jetson Nano용) 또는 onnxruntime")


class YOLOONNX:
    """YOLO ONNX Runtime 추론 클래스"""
    
    def __init__(self, onnx_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Args:
            onnx_path: ONNX 모델 파일 경로 (.onnx)
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime을 사용할 수 없습니다. pip install onnxruntime-gpu")
        
        self.onnx_path = onnx_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # ONNX Runtime 세션 생성
        self.session = self._create_session()
        
        # 입력/출력 정보 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # 입력 크기 (YOLOv8n은 640x640)
        self.input_size = 640
        
        print(f"ONNX 모델 로드 완료: {onnx_path}")
        print(f"입력 크기: {self.input_shape}")
        
    def _create_session(self):
        """ONNX Runtime 세션 생성"""
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX 파일을 찾을 수 없습니다: {self.onnx_path}")
        
        # 사용 가능한 프로바이더 확인
        available_providers = ort.get_available_providers()
        print(f"사용 가능한 ONNX Runtime 프로바이더: {available_providers}")
        
        # 프로바이더 우선순위: CUDA (있으면) -> CPU
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        try:
            session = ort.InferenceSession(
                self.onnx_path,
                providers=providers
            )
            print(f"ONNX Runtime 세션 생성 완료")
            print(f"사용 중인 프로바이더: {session.get_providers()}")
        except Exception as e:
            print(f"프로바이더 설정 실패, CPU로 재시도: {e}")
            try:
                session = ort.InferenceSession(
                    self.onnx_path,
                    providers=['CPUExecutionProvider']
                )
                print(f"CPU 프로바이더로 세션 생성 완료")
            except Exception as e2:
                raise RuntimeError(f"ONNX Runtime 세션 생성 실패: {e2}")
        
        return session
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (리사이즈, 정규화)"""
        # 리사이즈 (640x640)
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-255 -> 0-1)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # HWC -> CHW 변환
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # 배치 차원 추가 (1, 3, 640, 640)
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch
    
    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict]:
        """
        YOLO 출력 후처리 (NMS, 좌표 변환)
        
        Args:
            output: 모델 출력 (1, num_classes+4, num_boxes) - YOLOv8 표준 형태
            original_shape: 원본 이미지 크기 (height, width)
        
        Returns:
            detections: 감지된 객체 리스트 [{'bbox': (x, y, w, h), 'conf': conf, 'class': class_id}]
        """
        # 출력 형태 확인 및 변환
        if len(output.shape) == 3:
            output = output[0]  # 배치 제거 (num_classes+4, num_boxes)
        
        # YOLOv8 출력 형태: (num_classes+4, num_boxes)
        # 예: (84, 8400) = (4 bbox + 80 classes, 8400 boxes)
        # 전치하여 (num_boxes, num_classes+4) 형태로 변환
        if output.shape[0] < output.shape[1]:
            # (num_classes+4, num_boxes) -> (num_boxes, num_classes+4)
            boxes = output.T
        else:
            # 이미 (num_boxes, num_classes+4) 형태
            boxes = output
        
        # 박스 필터링 (신뢰도 임계값)
        detections = []
        original_h, original_w = original_shape
        
        for box in boxes:
            # YOLOv8 출력: [x_center, y_center, width, height, class1_conf, class2_conf, ...]
            # 좌표는 0-1 정규화된 값
            x_center, y_center, width, height = box[:4]
            class_scores = box[4:]
            
            # 최대 신뢰도 클래스 찾기
            class_id = np.argmax(class_scores)
            confidence = float(class_scores[class_id])
            
            # 신뢰도 필터링
            if confidence < self.conf_threshold:
                continue
            
            # 좌표 변환 (0-1 정규화 -> 픽셀 좌표)
            # YOLOv8 출력은 이미 원본 이미지 크기에 맞춰 정규화되어 있음
            x_center_px = x_center * original_w
            y_center_px = y_center * original_h
            width_px = width * original_w
            height_px = height * original_h
            
            # 중심 좌표 -> 좌상단 좌표
            x = int(x_center_px - width_px / 2)
            y = int(y_center_px - height_px / 2)
            w = int(width_px)
            h = int(height_px)
            
            # 좌표 범위 제한
            x = max(0, min(x, original_w - 1))
            y = max(0, min(y, original_h - 1))
            w = max(1, min(w, original_w - x))
            h = max(1, min(h, original_h - y))
            
            detections.append({
                'bbox': (x, y, w, h),
                'center': (int(x_center_px), int(y_center_px)),
                'conf': confidence,
                'class': int(class_id),
                'area': w * h
            })
        
        # NMS 적용
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Non-Maximum Suppression 적용"""
        if len(detections) == 0:
            return []
        
        # 박스와 신뢰도 추출
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['conf'] for d in detections])
        
        # OpenCV NMS 사용
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # 선택된 박스만 반환
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        
        return [detections[i] for i in indices]
    
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 객체 감지
        
        Args:
            image: 입력 이미지 (BGR)
        
        Returns:
            detections: 감지된 객체 리스트
        """
        try:
            original_shape = image.shape[:2]  # (height, width)
            
            # 전처리
            input_data = self.preprocess(image)
            
            # ONNX Runtime 추론
            outputs = self.session.run([self.output_name], {self.input_name: input_data})
            output = outputs[0]
            
            # 출력 형태 디버깅 (필요시)
            # print(f"출력 형태: {output.shape}")
            
            # 후처리
            detections = self.postprocess(output, original_shape)
            
            return detections
        except Exception as e:
            print(f"YOLO 추론 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
