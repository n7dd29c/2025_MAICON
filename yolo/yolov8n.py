import os
import glob
import sys
import subprocess

# ------------------------------------------------
# 1) 필수 라이브러리 확인 (ultralytics / torch)
# ------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    print("[INFO] ultralytics가 설치되어 있지 않습니다. pip로 설치를 진행합니다.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

try:
    import torch
except ImportError:
    print("[INFO] torch가 설치되어 있지 않습니다. pip로 설치를 진행합니다.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    import torch

import yaml  # data.yaml 읽기용


# ------------------------------------------------
# 2) 환경 및 경로 설정
# ------------------------------------------------
# TODO: 본인 환경에 맞게 수정
DATASET_PATH = r"./Object7.v7-maicon_mortar_background.yolov8"
SAVE_DIR = r"./Object7.v7-maicon_mortar_background.yolov8"

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL_NAME = "yolov8n"  # 가중치 파일 이름 및 러닝 이름


# ------------------------------------------------
# 3) 시스템 정보 출력
# ------------------------------------------------
def print_system_info():
    print("==== 환경 정보 ====")
    print("PyTorch 버전:", torch.__version__)
    print("CUDA 사용 가능?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA 버전:", torch.version.cuda)
        print("GPU 이름:", torch.cuda.get_device_name(0))
    else:
        print("현재는 CPU 모드로 동작 중입니다.")
    print("현재 작업 경로:", os.getcwd())
    print("===================")


# ------------------------------------------------
# 4) 데이터셋 / YAML 경로 찾기
# ------------------------------------------------
def find_dataset_yaml(dataset_path: str) -> str:
    print(f"DATASET_PATH: {dataset_path}")
    yaml_candidates = glob.glob(os.path.join(dataset_path, "**", "data.yaml"), recursive=True)
    assert len(yaml_candidates) > 0, "data.yaml을 찾지 못했습니다. 데이터셋 경로를 다시 확인하세요."
    data_yaml = yaml_candidates[0]
    print("사용할 data.yaml:", data_yaml)
    return data_yaml


# ------------------------------------------------
# 5) YOLOv8n 학습
# ------------------------------------------------
def train_yolov8n(data_yaml: str, save_dir: str) -> str:
    """
    data.yaml을 이용해 yolov8n 모델을 학습하고,
    best.pt 가중치 경로를 반환합니다.
    """
    # YOLO("yolov8n.pt")는 로컬에 없으면 자동으로 다운로드합니다.
    model = YOLO("yolov8n.pt")

    print(f"[INFO] 학습 시작 (epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH_SIZE})")
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=save_dir,
        name=MODEL_NAME,
    )

    # 기본적으로 runs/train/ 하위에 저장되지만,
    # 여기서는 프로젝트/이름 구조를 SAVE_DIR/ MODEL_NAME 로 가정합니다.
    weights_dir = os.path.join(save_dir, MODEL_NAME, "weights")
    best_ckpt = os.path.join(weights_dir, "best.pt")
    last_ckpt = os.path.join(weights_dir, "last.pt")

    if os.path.exists(best_ckpt):
        print("[INFO] best.pt 발견:", best_ckpt)
        return best_ckpt
    elif os.path.exists(last_ckpt):
        print("[INFO] best.pt 를 찾지 못하여 last.pt 사용:", last_ckpt)
        return last_ckpt
    else:
        raise FileNotFoundError("best.pt 및 last.pt 가중치를 찾지 못했습니다.")


# ------------------------------------------------
# 6) 간단 추론 테스트
# ------------------------------------------------
def inference_test(ckpt_path: str, dataset_path: str, max_images: int = 50):
    """
    학습된 ckpt_path로 valid 이미지 일부를 추론합니다.
    """
    print("[INFO] 추론에 사용할 가중치:", ckpt_path)
    if not os.path.exists(ckpt_path):
        print("[WARN] 체크포인트를 찾지 못했습니다. 추론을 건너뜁니다.")
        return

    model = YOLO(ckpt_path)

    # data.yaml 구조를 기반으로 일반적인 valid 이미지 경로 예시
    valid_img_dir = os.path.join(dataset_path, "**", "valid", "images", "*.jpg")
    test_imgs = glob.glob(valid_img_dir, recursive=True)[:max_images]

    if not test_imgs:
        print("[WARN] 테스트 이미지가 없습니다. 추론을 스킵합니다.")
        return

    print(f"[INFO] {len(test_imgs)}장의 이미지를 대상으로 추론을 수행합니다.")
    preds = model.predict(test_imgs, save=True)
    print("[INFO] 추론 완료. runs/predict 또는 프로젝트 폴더 결과를 확인하세요.")


# ------------------------------------------------
# 7) ONNX 내보내기 (TensorRT 변환 X)
# ------------------------------------------------
def export_to_onnx(ckpt_path: str, opset: int = 12, dynamic: bool = False) -> str:
    """
    학습된 ckpt(.pt)를 ONNX로 내보냅니다.
    반환값: 생성된 ONNX 파일 경로
    """
    print("[INFO] ONNX 내보내기 시작")
    model = YOLO(ckpt_path)

    export_kwargs = {
        "format": "onnx",
        "opset": opset,
        "dynamic": dynamic,  # 호출 시 dynamic=True/False 선택 가능
        "imgsz": IMG_SIZE,
        # 필요하면 simplify=True 등 옵션 추가 가능
    }

    onnx_path = model.export(**export_kwargs)
    # ultralytics 8.x 기준, export()는 경로 문자열을 반환합니다.
    print("[INFO] ONNX 내보내기 완료:", onnx_path)
    return onnx_path


# ------------------------------------------------
# 8) main 함수 (TensorRT 변환 호출 제거)
# ------------------------------------------------
def main():
    print_system_info()

    # 1) data.yaml 찾기
    data_yaml = find_dataset_yaml(DATASET_PATH)

    # 2) YOLOv8n 학습
    ckpt_path = train_yolov8n(data_yaml, SAVE_DIR)

    # 3) 간단 추론 테스트
    inference_test(ckpt_path, DATASET_PATH, max_images=50)

    # 4) ONNX 내보내기만 수행 (TensorRT 변환 X)
    # dynamic=True/False는 상황에 맞게 선택해서 쓰시면 됩니다.
    onnx_path = export_to_onnx(ckpt_path, opset=12, dynamic=False)
    print("[INFO] 최종 ONNX 경로:", onnx_path)


if __name__ == "__main__":
    main()
