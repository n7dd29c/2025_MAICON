import os
import glob
from ultralytics import YOLO


# ------------------------------------------------
# 1) 가장 최근 best.pt 파일 찾기
# ------------------------------------------------
def find_latest_best_pt(weights_dir: str) -> str:
    best_files = glob.glob(os.path.join(weights_dir, "best*.pt"))

    if not best_files:
        raise FileNotFoundError(f"{weights_dir} 내부에 best.pt 파일이 없습니다.")

    def extract_number(filename):
        base = os.path.basename(filename)
        number_part = base.replace("best", "").replace(".pt", "")
        return int(number_part) if number_part.isdigit() else 0

    latest_file = max(best_files, key=extract_number)
    print(f"[INFO] 최신 best.pt 파일: {latest_file}")
    return latest_file


# ------------------------------------------------
# 2) ONNX 변환 함수
# ------------------------------------------------
def export_to_onnx(ckpt_path: str, opset: int = 12, dynamic: bool = False) -> str:
    print("[INFO] ONNX 내보내기 시작:", ckpt_path)

    model = YOLO(ckpt_path)

    export_kwargs = {
        "format": "onnx",
        "opset": opset,
        "dynamic": dynamic,
        "imgsz": 640,
    }

    onnx_path = model.export(**export_kwargs)
    print("[INFO] ONNX 내보내기 완료:", onnx_path)
    return onnx_path


# ------------------------------------------------
# 3) main
# ------------------------------------------------
def main():
    weights_dir = r"C:\Users\Admin\Desktop\2025_MAICON\yolo\Object7.v7-maicon_mortar_background.yolov8\yolov8n18\weights"

    latest_best_pt = find_latest_best_pt(weights_dir)

    onnx_path = export_to_onnx(latest_best_pt, opset=12, dynamic=False)

    print("[INFO] 최종 ONNX 파일:", onnx_path)


if __name__ == "__main__":
    main()
