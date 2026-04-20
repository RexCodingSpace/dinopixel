import torch
import os
import sys
import argparse
import numpy as np
import onnx
from onnxsim import simplify
from contextlib import ExitStack

# 確保能匯入你的網路架構
from networks.PixelFormer import MambaPixelFormer


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def get_args():
    parser = argparse.ArgumentParser(description='PixelFormer ONNX Export', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--encoder', type=str, default='large07')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--dinov2_model', type=str, default='dinov2_vitl14')
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--input_width', type=int, default=640)
    parser.add_argument('--max_depth', type=float, default=10)
    parser.add_argument('--output_onnx', type=str, default='mamba_pixelformer.onnx')

    if sys.argv.__len__() == 2 and sys.argv[1].startswith('@'):
        args, unknown = parser.parse_known_args([sys.argv[1]])
    else:
        args, unknown = parser.parse_known_args()

    if unknown:
        print(f"== [Info] Ignoring unknown arguments in config: {unknown}")

    return args


def no_autocast_context():
    """同時關閉 CPU 與 CUDA autocast 的 context manager (雙重保險)"""
    stack = ExitStack()
    stack.enter_context(torch.no_grad())
    stack.enter_context(torch.amp.autocast(device_type='cpu', enabled=False))
    if torch.cuda.is_available():
        stack.enter_context(torch.amp.autocast(device_type='cuda', enabled=False))
    return stack


def main():
    args = get_args()
    device = torch.device('cpu')

    # --- 關鍵修正 1: 強制設定預設型別 ---
    torch.set_default_dtype(torch.float32)

    # --- 關鍵修正 2: 初始化時關閉 AMP (use_amp=False) ---
    # MambaPixelFormer 內部有硬編碼的 autocast(bfloat16),
    # 關掉才能避免 forward 時自動把 tensor cast 成 BF16
    print(f"== Initializing MambaPixelFormer with {args.dinov2_model} (AMP disabled for export)...")
    model = MambaPixelFormer(
        dinov2_model=args.dinov2_model,
        min_depth=0.1,
        max_depth=args.max_depth,
        use_amp=False,
    )

    # --- 載入 checkpoint 權重 ---
    print(f"== Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)

    # 把 state_dict 裡所有 bf16/fp16 都轉成 fp32 再載入
    state_dict = {
        k: (v.float() if isinstance(v, torch.Tensor) and v.dtype in (torch.bfloat16, torch.float16) else v)
        for k, v in state_dict.items()
    }
    # 處理 DDP 的 module. 前綴
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [Warning] Missing keys: {len(missing)} (showing first 5) {missing[:5]}")
    if unexpected:
        print(f"  [Warning] Unexpected keys: {len(unexpected)} (showing first 5) {unexpected[:5]}")

    # --- 關鍵修正 3: 深度強制轉型 ---
    print("== Forcing model to Float32 (deep cast)...")
    model = model.to(device)

    for param in model.parameters():
        param.data = param.data.float()
        if param.grad is not None:
            param.grad.data = param.grad.data.float()

    for buf_name, buf in list(model.named_buffers()):
        if buf.dtype in (torch.bfloat16, torch.float16):
            module_path, _, attr_name = buf_name.rpartition('.')
            mod = model.get_submodule(module_path) if module_path else model
            setattr(mod, attr_name, buf.float())

    model.float()
    model.eval()

    # 虛擬輸入
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width, dtype=torch.float32, device=device)

    # ========== DIAGNOSTIC: Forward hook trace ==========
    print("=" * 60)
    print("DIAGNOSTIC: Sanity check with autocast disabled")
    print("=" * 60)

    bf16_hits = []

    def make_hook(name):
        def hook(module, inputs, output):
            def check_tensor(t, tag):
                if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16:
                    bf16_hits.append(f"[{tag}] {name} ({type(module).__name__})")

            if isinstance(inputs, tuple):
                for i, inp in enumerate(inputs):
                    check_tensor(inp, f"INPUT[{i}]")
            if isinstance(output, torch.Tensor):
                check_tensor(output, "OUTPUT")
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    check_tensor(o, f"OUTPUT[{i}]")
            elif isinstance(output, dict):
                for k, v in output.items():
                    check_tensor(v, f"OUTPUT[{k}]")
        return hook

    handles = []
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    with no_autocast_context():
        try:
            test_out = model(dummy_input)
            if isinstance(test_out, torch.Tensor):
                print(f"  [sanity check] output dtype = {test_out.dtype}, shape = {tuple(test_out.shape)}")
            else:
                print(f"  [sanity check] output type = {type(test_out)}")
        except Exception as e:
            print(f"  [Forward failed] {e}")
            import traceback
            traceback.print_exc()
            return

    for h in handles:
        h.remove()

    print(f"\n[BFloat16 hits] ({len(bf16_hits)} found):")
    for h in bf16_hits[:10]:
        print(f"  {h}")
    if len(bf16_hits) > 10:
        print(f"  ... and {len(bf16_hits) - 10} more")

    if len(bf16_hits) == 0:
        print("  ✓ All tensors are FP32. Safe to export.")
    else:
        print("  ✗ Still have BF16 tensors. Export may fail.")
    print("=" * 60 + "\n")
    # ========== END DIAGNOSTIC ==========

    # --- ONNX Export ---
    print(f"== Exporting to ONNX (Opset 14)...")
    try:
        with no_autocast_context():
            torch.onnx.export(
                model,
                dummy_input,
                args.output_onnx,
                export_params=True,
                opset_version=17,
                do_constant_folding=False,
                input_names=['input'],
                output_names=['output'],
                #dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        print(f"== Export successful: {args.output_onnx}")
    except Exception as e:
        print(f"== [Export Failed] {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 簡化步驟 ---
    print(f"== Simplifying ONNX model...")
    try:
        model_onnx = onnx.load(args.output_onnx)
        model_simp, check = simplify(model_onnx)
        if check:
            sim_path = args.output_onnx.replace('.onnx', '_sim.onnx')
            onnx.save(model_simp, sim_path)
            print(f"== Simplified model saved as: {sim_path}")
            target_path = sim_path
        else:
            print(f"== [Warning] Simplification check failed, using original.")
            target_path = args.output_onnx
    except Exception as e:
        print(f"== [Warning] Simplification failed: {e}")
        target_path = args.output_onnx

    # --- 數值驗證 ---
    print(f"== Verifying numerical consistency using {target_path}...")
    import onnxruntime as ort

    with no_autocast_context():
        torch_out = model(dummy_input).detach().cpu().numpy()

    ort_session = ort.InferenceSession(target_path, providers=['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    diff = np.abs(torch_out - ort_out).mean()
    max_diff = np.abs(torch_out - ort_out).max()
    print(f"== Mean absolute difference: {diff:.6f}")
    print(f"== Max  absolute difference: {max_diff:.6f}")

    if diff < 1e-3:
        print("== ✓ Numerical consistency: PASS")
    else:
        print("== ✗ Numerical consistency: WARNING (diff > 1e-3)")


if __name__ == '__main__':
    main()