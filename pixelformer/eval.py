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
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
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
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                continue

            pred_depth = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
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

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu


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
    model = MambaPixelFormer(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    
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
            
        model.load_state_dict(new_state_dict)
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