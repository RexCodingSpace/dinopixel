import torch
import time
import numpy as np

# ==============================================================================
# 1. 這裡要對應你的檔案名稱
#    假設你的模型檔案叫做 networks/PixelFormer.py
#    我們引入你在裡面定義的便捷函數 (Helper Functions)
# ==============================================================================
try:
    from networks.PixelFormer import MambaPixelFormer
except ImportError as e:
    print("❌ Import Error: 找不到模型檔案。請確認 networks/PixelFormer.py 存在。")
    print(f"詳細錯誤: {e}")
    exit()

# 嘗試引入 thop 來計算 FLOPs (非必要，沒有會自動跳過)
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

def measure_inference_speed(model, input_shape=(1, 3, 480, 640), device='cuda'):
    """
    測量 PyTorch 模型的推論速度 (含預熱與 CUDA 同步)
    """
    print(f"🚀 開始測速: {model.__class__.__name__}")
    print(f"   Input Shape: {input_shape}")
    print(f"   Device: {device}")
    
    # 1. 準備輸入資料 (模擬一張隨機圖片)
    dummy_input = torch.randn(input_shape).to(device)

    # 2. 準備模型
    model.to(device)
    model.eval()  # 切換到評估模式

    # ------------------------------------------------------------------
    # [FLOPs 計算]
    # ------------------------------------------------------------------
    if HAS_THOP:
        print("\n[*] 正在計算 FLOPs 和 參數量 (可能會花幾秒鐘)...")
        try:
            # thop 運算時模型和輸入要在同一裝置
            macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
            print(f"    📦 參數量 (Params)   : {params / 1e6:.2f} M (百萬)")
            print(f"    🧮 運算量 (GFLOPs)   : {macs / 1e9:.2f} G (十億)")
        except Exception as e:
            print(f"    ⚠️ FLOPs 計算失敗 (可能是自定義層不支援): {e}")
    else:
        print("\n[!] 未安裝 thop，跳過 FLOPs 計算 (pip install thop)")

    print("-" * 60)

    # 3. 預熱 (Warm-up)
    # 讓 GPU 載入 CUDA Kernels，前幾次通常會慢，不計入統計
    print("[*] 正在預熱 GPU (Warm-up 50 次)...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    torch.cuda.synchronize() # 等待 GPU 跑完

    # 4. 正式測量
    print("[*] 正式測速 (迴圈 100 次)...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(100):
            starter.record()
            
            # === 模型推論 ===
            _ = model(dummy_input)
            # ================
            
            ender.record()
            torch.cuda.synchronize() # 等待 GPU 跑完
            
            curr_time = starter.elapsed_time(ender) # 毫秒 (ms)
            timings.append(curr_time)

    # 5. 統計結果
    avg_latency = np.mean(timings)
    std_latency = np.std(timings)
    fps = 1000 / avg_latency

    print("-" * 60)
    print(f"✅ 測速結果 ({torch.cuda.get_device_name(0)})")
    print(f"   平均延遲 (Latency): {avg_latency:.2f} ms ± {std_latency:.2f} ms")
    print(f"   每秒幀數 (FPS)    : {fps:.2f}")
    print("-" * 60)
    
    return fps

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ 錯誤: 找不到 GPU (CUDA)，測速無意義。")
        exit()
    
    DEVICE = 'cuda'
    
    # ==========================================================================
    # 設定測試參數 (請依需求修改這裡)
    # ==========================================================================
    
    # 1. 選擇要測的模型版本
    # 選項: pixelformer_dinov2_vitb14 (Base), pixelformer_dinov2_vitl14 (Large)
    print("正在建立模型 (DINOv2)...")
    
    model = MambaPixelFormer()

    # 2. 設定解析度 (Batch, Channel, Height, Width)
    # 常見設定: (1, 3, 480, 640) 或 (1, 3, 352, 1216) 依資料集而定
    TEST_INPUT_SHAPE = (1, 3, 480, 640)

    # 開始執行
    measure_inference_speed(model, input_shape=TEST_INPUT_SHAPE, device=DEVICE)