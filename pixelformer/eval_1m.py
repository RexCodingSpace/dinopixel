import torch
import torch.backends.cudnn as cudnn

import os, sys
import argparse
import numpy as np
from tqdm import tqdm

from utils import post_process_depth, flip_lr, compute_errors, eval_metrics
from networks.PixelFormer import MambaPixelFormer

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='PixelFormer PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='pixelformer')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', required=True)
parser.add_argument('--dinov2_model',              type=str,   help='dino model: dinov2_vits14, dinov2_vitb14, dinov2_vitl14', default='dinov2_vitl14') 
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Preprocessing
parser.add_argument('--do_random_rotate',          help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                 help='if set, will randomly use right images when train on KITTI', action='store_true')

# Eval
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=True)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=True)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=True)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-1)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=10.0)
parser.add_argument('--eigen_crop',                help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                 help='if set, crops according to Garg  ECCV16', action='store_true')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader

def eval(model, dataloader_eval, post_process=False):
    # 初始化兩套指標：[0..8] 是指標，[9] 是計數器
    eval_all = torch.zeros(10).cuda()
    eval_1m = torch.zeros(10).cuda()

    print("== Starting Evaluation (Global vs. <1m) ==")
    a=0
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            a=a+1
            if a==1:
                print(gt_depth.flatten()[153600:153700])
            
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                continue

            # 模型推論
            pred_depth = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        # 1. 處理 KITTI Crop 邏輯 (與你原始流程一致)
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        # 2. 深度數值清理 (與你原始流程一致)
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        # 3. 建立有效遮罩 (包含 Eigen/Garg Crop)
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

        # --- A. 計算全域指標 (Overall) ---
        measures_all = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        eval_all[:9] += torch.tensor(measures_all).cuda()
        eval_all[9] += 1

        # --- B. 計算 1m 內指標 (<1m Only) ---
        mask_1m = np.logical_and(valid_mask, gt_depth <= 2)
        if mask_1m.sum() > 50: # 確保至少有足夠像素才計算，避免單點雜訊
            measures_1m = compute_errors(gt_depth[mask_1m], pred_depth[mask_1m])
            eval_1m[:9] += torch.tensor(measures_1m).cuda()
            eval_1m[9] += 1

    # --- 最終數據處理與列印 ---
    def print_result_table(title, results_tensor):
        cnt = results_tensor[9].item()
        if cnt == 0:
            print(f"\n[{title}] No valid samples found.")
            return
        
        final_metrics = results_tensor[:9].cpu() / cnt
        print(f"\n[{title}] (Samples: {int(cnt)})")
        header = "{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
            'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'
        )
        print(header)
        res_str = ", ".join(["{:7.4f}".format(final_metrics[i]) for i in range(9)])
        print(res_str)
        return final_metrics

    # 印出全域結果
    metrics_all = print_result_table("OVERALL PERFORMANCE", eval_all)
    # 印出 1m 內結果
    metrics_1m = print_result_table("NEAR-FIELD PERFORMANCE (< 1.0m)", eval_1m)

    return eval_all.cpu() / eval_all[9].item()

def main_worker(args):
    # 1. 初始化模型
    # 手動補上 dataloader.py 需要但 eval 沒用到的參數
    if not hasattr(args, 'distributed'):
        args.distributed = False
    if not hasattr(args, 'world_size'):
        args.world_size = 1
    if not hasattr(args, 'rank'):
        args.rank = 0
    if not hasattr(args, 'gpu'):
        args.gpu = 0
    model = MambaPixelFormer(dinov2_model = args.dinov2_model, min_depth= 0.1, max_depth=args.max_depth)
    
    # 2. 載入權重 (包含處理 module. 前綴)
    if os.path.isfile(args.checkpoint_path):
        print("== Loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        
        # 處理 DDP 訓練產生的 'module.' 前綴
        state_dict = checkpoint.get('model', checkpoint) # 兼容只存 state_dict 或存 dict 的情況
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") # 移除 module.
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print("== Loaded checkpoint successfully")
    else:
        print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        return

    # 3. 設定 GPU
    model.cuda()
    model.eval()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    cudnn.benchmark = True

    # 4. 載入資料
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # 5. 開始評估
    with torch.no_grad():
        import time
        torch.cuda.synchronize()
        start = time.time()

        # post_process=True 會開啟 Flip Testing (TTA)
        eval_measures = eval(model, dataloader_eval, post_process=False)

        torch.cuda.synchronize()
        end = time.time()

        total_time = end - start
        try:
            num_images = len(dataloader_eval.data.dataset)
        except Exception:
            num_images = len(dataloader_eval.data)

        print(f"\n[Overall Eval Time]")
        print(f"Total = {total_time:.3f}s")
        if num_images > 0:
            print(f"Per image = {total_time/num_images:.4f}s | FPS = {num_images/total_time:.2f}")


def main():
    torch.cuda.empty_cache()
    # 如果有多張 GPU，這行會確保程式只用第一張，避免 DDP 衝突
    if torch.cuda.device_count() > 1:
        print("Multiple GPUs detected. Using the first GPU for evaluation.")
        
    main_worker(args)


if __name__ == '__main__':
    main()