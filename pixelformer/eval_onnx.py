import torch
import torch.backends.cudnn as cudnn
import onnxruntime as ort
import os, sys
import argparse
import numpy as np
import time
from tqdm import tqdm

# 沿用你原本的 utils
from utils import post_process_depth, flip_lr, compute_errors, eval_metrics

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='PixelFormer ONNX Evaluation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

# 新增 ONNX 路徑參數
parser.add_argument('--onnx_path', type=str, help='path to the onnx model', default="mamba_pixelformer.onnx")

# 沿用原本的參數
parser.add_argument('--dataset', type=str, help='dataset, kitti or nyu', default='nyu')
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width',  default=640)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-1)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10.0)
parser.add_argument('--do_kb_crop', help='crop input images as kitti benchmark', action='store_true')
parser.add_argument('--eigen_crop', help='crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='crops according to Garg ECCV16', action='store_true')

# Eval Data Path
parser.add_argument('--data_path_eval', type=str, help='path to data', required=True)
parser.add_argument('--gt_path_eval', type=str, help='path to groundtruth', required=True)
parser.add_argument('--filenames_file_eval', type=str, help='path to filenames file', required=True)

# 為了相容 NewDataLoader 內部邏輯，保留這些不影響 ONNX 的參數
parser.add_argument('--do_random_rotate', action='store_true')
parser.add_argument('--degree', type=float, default=2.5)
parser.add_argument('--use_right', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args, unknown = parser.parse_known_args([arg_filename_with_prefix])
else:
    args, unknown = parser.parse_known_args()

# 載入對應的 DataLoader
if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader

def eval_onnx(ort_session, dataloader_eval):
    eval_measures = torch.zeros(10).cuda()
    input_name = ort_session.get_inputs()[0].name
    
    # 這裡開始計時
    torch.cuda.synchronize()
    start_time = time.time()

    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        # 1. 取得影像並轉為 NumPy (ONNX 格式)
        image_tensor = eval_sample_batched['image']
        image_np = image_tensor.numpy()
        
        gt_depth = eval_sample_batched['depth']
        has_valid_depth = eval_sample_batched['has_valid_depth']
        if not has_valid_depth:
            continue

        # 2. ONNX 推論
        # ort_session.run 回傳的是一個 list
        pred_depth = ort_session.run(None, {input_name: image_np})[0]
        
        # 3. 後處理 (與你原本的 eval.py 邏輯一致)
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)

        # 4. 計算誤差
        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    torch.cuda.synchronize()
    end_time = time.time()

    # 5. 輸出統計結果
    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    
    print('\n' + '='*50)
    print(f'ONNX Evaluation Results ({int(cnt)} samples)')
    print('='*50)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))
    
    total_time = end_time - start_time
    print('-'*50)
    print(f"Total Time: {total_time:.3f}s")
    print(f"FPS: {cnt/total_time:.2f}")
    print('='*50)

    return eval_measures_cpu

def main():
    # 1. 初始化 ONNX Session
    # 如果你在 Orin 上，可以把 providers 改成 ['CUDAExecutionProvider']
    print(f"== Initializing ONNX Runtime with: {args.onnx_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(args.onnx_path, providers=providers)

    # --- 加入這段診斷代碼 ---
    active_providers = ort_session.get_providers()
    # 這是正確的查詢方式
    print(f"== [Diagnostic] All available providers in env: {ort.get_available_providers()}")
    print(f"== [Diagnostic] Active providers for this session: {ort_session.get_providers()}")

    if 'CUDAExecutionProvider' not in active_providers:
        print("== [WARNING] CUDA is NOT being used! Fallback to CPU.")
    # ----------------------

    # 2. 準備 DataLoader
    # 補上必要的 args 屬性以相容 NewDataLoader
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # 3. 執行評估
    eval_onnx(ort_session, dataloader_eval)

if __name__ == '__main__':
    main()