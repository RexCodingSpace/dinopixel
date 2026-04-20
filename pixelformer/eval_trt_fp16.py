import torch
import torch.nn.functional as F
import tensorrt as trt
import numpy as np
import os, sys
import argparse
import time
from tqdm import tqdm

# 載入實驗室工具 (請確保 utils.py 和 dataloaders 在同目錄下)
from utils import post_process_depth, compute_errors
from dataloaders.dataloader import NewDataLoader

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip(): continue
        yield arg

def get_args():
    parser = argparse.ArgumentParser(description='PixelFormer TRT 8.6 Eval', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    
    # 預設指向你在 8.6.1 建好的 engine
    parser.add_argument('--engine', type=str, default="mamba_trt8.engine", help='TRT 8.6 engine path')
    parser.add_argument('--dataset', type=str, default='nyu')
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--input_width', type=int, default=640)
    parser.add_argument('--min_depth_eval', type=float, default=1e-1)
    parser.add_argument('--max_depth_eval', type=float, default=10.0)
    parser.add_argument('--data_path_eval', type=str, required=True)
    parser.add_argument('--gt_path_eval', type=str, required=True)
    parser.add_argument('--filenames_file_eval', type=str, required=True)
    parser.add_argument('--do_kb_crop', action='store_true')
    parser.add_argument('--eigen_crop', action='store_true')
    parser.add_argument('--garg_crop', action='store_true')
    
    # DataLoader 相容參數
    parser.add_argument('--do_random_rotate', action='store_true')
    parser.add_argument('--degree', type=float, default=2.5)
    parser.add_argument('--use_right', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args, unknown = parser.parse_known_args([arg_filename_with_prefix])
    else:
        args, unknown = parser.parse_known_args()
    return args

class TensorRTInference:
    def __init__(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}. Please check TRT version compatibility.")
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        # --- TRT 8.6 核心改動：使用 Binding 邏輯 ---
        self.bindings = [None] * self.engine.num_bindings
        self.buffers = {}
        self.input_idx = -1
        self.output_name = None

        print("== [Inference Info] Binding Tensors (TRT 8.6 Style):")
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            if self.engine.binding_is_input(i):
                self.input_idx = i
                print(f"   [Input]  {name}: shape={shape}")
            else:
                # 建立輸出 Buffer
                self.output_name = name
                out_tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
                self.buffers[name] = out_tensor
                self.bindings[i] = out_tensor.data_ptr()
                print(f"   [Output] {name}: shape={shape}")

    def run(self, image_tensor):
        # 確保輸入連續並綁定地址
        image_tensor = image_tensor.contiguous()
        self.bindings[self.input_idx] = image_tensor.data_ptr()
        
        # 執行推論 (TRT 8.6 使用 execute_async_v2)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.cuda_stream
        )
        self.stream.synchronize()
        
        # 抓取最終深度圖
        return self.buffers[self.output_name]

def main():
    args = get_args()
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    
    # 1. 準備數據與模型
    dataloader = NewDataLoader(args, 'online_eval')
    model = TensorRTInference(args.engine)
    
    eval_measures = torch.zeros(10).cuda()
    
    print(f"\n== [Start] Running Evaluation: {args.engine}")
    start_time = time.time()

    # 2. 評估迴圈
    for i, sample in enumerate(tqdm(dataloader.data)):
        with torch.no_grad():
            img = sample['image'].cuda() 
            gt_depth = sample['depth'].squeeze().numpy()
            if not sample['has_valid_depth']: continue

            # 對齊 Engine 靜態輸入 (480x640)
            if img.shape[2:] != (args.input_height, args.input_width):
                img_input = F.interpolate(img, size=(args.input_height, args.input_width), 
                                         mode='bilinear', align_corners=True)
            else:
                img_input = img

            # 3. 推論
            pred_trt = model.run(img_input)
            
            # Resize 回原圖大小算誤差
            if pred_trt.shape[-2:] != gt_depth.shape:
                pred_trt = F.interpolate(pred_trt, size=gt_depth.shape, 
                                        mode='bilinear', align_corners=True)
            
            pred_depth = pred_trt.cpu().numpy().squeeze()

            # 4. 後處理 (Clip & Crop)
            pred_depth = np.clip(pred_depth, args.min_depth_eval, args.max_depth_eval)
            valid_mask = (gt_depth > args.min_depth_eval) & (gt_depth < args.max_depth_eval)

            if args.do_kb_crop:
                h, w = gt_depth.shape
                top, left = int(h - 352), int((w - 1216) / 2)
                full_pred = np.zeros((h, w), dtype=np.float32)
                full_pred[top:top+352, left:left+1216] = pred_depth
                pred_depth = full_pred

            if args.garg_crop or args.eigen_crop:
                gh, gw = gt_depth.shape
                mask = np.zeros_like(valid_mask)
                if args.garg_crop:
                    mask[int(0.408*gh):int(0.991*gh), int(0.035*gw):int(0.964*gw)] = 1
                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        mask[int(0.332*gh):int(0.913*gh), int(0.035*gw):int(0.964*gw)] = 1
                    elif args.dataset == 'nyu':
                        mask[45:471, 41:601] = 1
                valid_mask = valid_mask & mask

            # 5. 計算指標
            errors = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:9] += torch.tensor(errors).cuda()
            eval_measures[9] += 1

    # 6. 統計與輸出
    total_time = time.time() - start_time
    cnt = eval_measures[9].item()
    if cnt == 0:
        print("Error: No valid images found for evaluation.")
        return

    final_res = (eval_measures[:9] / cnt).cpu().numpy()
    
    print("\n" + "="*70)
    print(f"Results for TensorRT 8.6.1 Engine")
    print(f"Total Images: {int(cnt)} | Avg FPS: {cnt / total_time:.2f}")
    print("-" * 70)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    print(", ".join([f"{x:7.4f}" for x in final_res]))
    print("="*70)

if __name__ == '__main__':
    main()