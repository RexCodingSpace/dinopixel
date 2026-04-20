import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine_trt8(onnx_path="mamba_pixelformer_sim.onnx", engine_path="mamba_trt8.engine"):
    # 1. 建立 Builder 和 Network
    builder = trt.Builder(TRT_LOGGER)
    # TRT 8.x 建立網路的標準寫法
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 2. 解析 ONNX (原始版本)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[Parser Error]: {parser.get_error(i)}")
            return

    # 3. 設定 FP16 與 Workspace
    if builder.platform_has_fast_fp16:
        print("== TRT 8.6.1: Enabling FP16 Mode...")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # TRT 8.x 設定 Workspace 的方式 (舊版 API 使用 max_workspace_size)
    # 這裡我們設定 4GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4GB

    # 4. 設定 Optimization Profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1,3,480,640), (1,3,480,640), (1,3,480,640))
    config.add_optimization_profile(profile)

    print("== Building Serialized Engine (TRT 10.x)...")
    plan = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"== Success! Saved to {engine_path}")

if __name__ == "__main__":
    build_engine_trt8()