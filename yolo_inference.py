# -*- coding: utf-8 -*-
# YOLO TensorRT 추론 모듈
# Jetson Nano 최적화된 YOLO 추론
import sys
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

# TensorRT 경로 추가 (Python 3.6 경로)
sys.path.insert(0, '/usr/lib/python3.6/dist-packages')

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_AVAILABLE = False
    import sys
    python_version = sys.version_info
    print(f"경고: TensorRT를 사용할 수 없습니다.")
    print(f"  Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"  오류: {e}")
    print(f"  해결 방법:")
    print(f"    1. Python 3.6으로 실행: python3.6 -m autonomous_driving")
    print(f"    2. pycuda 설치 확인: pip3 install pycuda")
    print(f"    3. TensorRT 경로 확인: ls /usr/lib/python3.6/dist-packages/tensorrt")


class YOLOTensorRT:
    """YOLO TensorRT 추론 클래스"""
    
    def __init__(self, engine_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Args:
            engine_path: TensorRT 엔진 파일 경로 (.trt)
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT를 사용할 수 없습니다.")
        
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # TensorRT 엔진 로드
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # 입력/출력 버퍼 설정
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # 입력 크기 (YOLOv8n은 640x640)
        self.input_size = 640
        
    def _load_engine(self):
        """TensorRT 엔진 로드"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            raise RuntimeError(f"엔진 로드 실패: {self.engine_path}")
        
        print(f"TensorRT 엔진 로드 완료: {self.engine_path}")
        return engine
    
    def _allocate_buffers(self):
        """입력/출력 버퍼 할당"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # GPU 메모리 할당
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
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
            output: 모델 출력 (1, num_classes+4, num_boxes) 또는 (1, num_boxes, num_classes+4)
            original_shape: 원본 이미지 크기 (height, width)
        
        Returns:
            detections: 감지된 객체 리스트 [{'bbox': (x, y, w, h), 'conf': conf, 'class': class_id}]
        """
        # 출력 형태: (1, num_classes+4, num_boxes) 또는 (1, num_boxes, num_classes+4)
        # YOLOv8 출력 형태에 맞게 처리
        if len(output.shape) == 3:
            output = output[0]  # 배치 제거
        
        # 출력 형태 확인 및 변환
        if output.shape[0] > output.shape[1]:
            # (num_boxes, num_classes+4) 형태
            boxes = output
        else:
            # (num_classes+4, num_boxes) 형태 -> 전치
            boxes = output.T
        
        # 박스 필터링 (신뢰도 임계값)
        detections = []
        original_h, original_w = original_shape
        
        for box in boxes:
            # YOLOv8 출력: [x_center, y_center, width, height, class1_conf, class2_conf, ...]
            x_center, y_center, width, height = box[:4]
            class_scores = box[4:]
            
            # 최대 신뢰도 클래스 찾기
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # 신뢰도 필터링
            if confidence < self.conf_threshold:
                continue
            
            # 좌표 변환 (0-1 정규화 -> 픽셀 좌표)
            # 원본 이미지 크기로 스케일링
            scale_x = original_w / self.input_size
            scale_y = original_h / self.input_size
            
            x_center_px = x_center * original_w
            y_center_px = y_center * original_h
            width_px = width * original_w
            height_px = height * original_h
            
            # 중심 좌표 -> 좌상단 좌표
            x = int(x_center_px - width_px / 2)
            y = int(y_center_px - height_px / 2)
            w = int(width_px)
            h = int(height_px)
            
            detections.append({
                'bbox': (x, y, w, h),
                'center': (int(x_center_px), int(y_center_px)),
                'conf': float(confidence),
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
        original_shape = image.shape[:2]  # (height, width)
        
        # 전처리
        input_data = self.preprocess(image)
        
        # GPU로 전송
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 추론 실행
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 결과 가져오기
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # 출력 형태 변환
        output = self.outputs[0]['host']
        output_shape = self.engine.get_binding_shape(1)  # 출력 바인딩 인덱스
        output = output.reshape(output_shape)
        
        # 후처리
        detections = self.postprocess(output, original_shape)
        
        return detections
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.free()

