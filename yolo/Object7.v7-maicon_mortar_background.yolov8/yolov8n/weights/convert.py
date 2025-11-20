# convert_onnx_to_trt.py
import tensorrt as trt
import sys
sys.path.insert(0, '/usr/lib/python3.6/dist-packages')

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # ONNX 파일 읽기
    print(f"Loading ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("ONNX file parsed successfully")
    
    # 빌더 설정
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        else:
            print("WARNING: FP16 not supported on this platform")
    
    # 엔진 빌드
    print("Building TensorRT engine. This may take a while...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("Failed to build engine")
        return None
    
    # 엔진 저장
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    print("Done! Engine saved successfully.")
    return engine

if __name__ == "__main__":
    import os
    
    # 현재 디렉토리에서 best.onnx 찾기
    onnx_path = "best.onnx"
    if not os.path.exists(onnx_path):
        # 상위 디렉토리에서 찾기
        onnx_path = "../weights/best.onnx"
        if not os.path.exists(onnx_path):
            print(f"ERROR: {onnx_path} not found")
            sys.exit(1)
    
    trt_path = onnx_path.replace('.onnx', '.trt')
    
    print(f"Converting {onnx_path} to {trt_path}")
    build_engine(onnx_path, trt_path, fp16_mode=True)