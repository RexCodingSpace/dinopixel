from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from contextlib import nullcontext
from tqdm import tqdm

# 匯入你的模型
from networks.PixelFormer import MambaPixelFormer 


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg
# 設定 Colormap
cmap_color = plt.get_cmap('inferno') 

def main():
    parser = argparse.ArgumentParser(description='MambaPixelFormer Video Inference', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    # 核心模型參數
    parser.add_argument('--model_name',      type=str,   default='mamba_pixelformer')
    parser.add_argument('--dinov2_model',    type=str,   default='dinov2_vitl14')
    parser.add_argument('--max_depth',       type=float, default=10.0)
    parser.add_argument('--checkpoint_path', type=str,   required=True)
    parser.add_argument('--input_height',    type=int,   default=480)
    parser.add_argument('--input_width',     type=int,   default=640)
    
    # 影片專用參數 (取代了原本的 data_path_eval 等)
    parser.add_argument('--video_path',      type=str,   required=True, help='當前目錄下的影片檔名或路徑')
    parser.add_argument('--output_path',     type=str,   default='output_depth.mp4', help='輸出影片的路徑')
    parser.add_argument('--use_amp',         action='store_true', default=True)

    args, unknown = parser.parse_known_args()

    # 1. 載入模型
    print(f"[Init] Loading {args.model_name} with {args.dinov2_model}...")
    model = MambaPixelFormer(
        dinov2_model=args.dinov2_model,
        max_depth=args.max_depth,
        use_amp=args.use_amp
    )
    
    # 載入權重並處理命名空間
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.eval()
    model.cuda()

    # 2. 開啟影片串流
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video_path}")
        return

    # 獲取影片屬性以建立 VideoWriter
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定輸出的影片編碼與路徑 (輸出解析度採用模型輸入大小)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (args.input_width, args.input_height))

    # ImageNet 標準化參數
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    print(f"=> Processing video: {args.video_path}")
    print(f"=> Total frames: {total_frames} | Target resolution: {args.input_width}x{args.input_height}")

    # 3. 逐幀處理
    pbar = tqdm(total=total_frames)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- 前處理 ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (args.input_width, args.input_height))
            
            # Normalization (0-1 並減去均值)
            img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
            input_tensor = torch.from_numpy(img_normalized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda()

            # --- 推論 ---
            with torch.no_grad():
                with model._autocast_context(input_tensor.device):
                    depth = model(input_tensor)
            
            # --- 後處理與視覺化 ---
            depth_pred = depth.cpu().squeeze().numpy()
            
            # 歸一化深度值 (0-1) 用於映射顏色
            depth_norm = np.clip(depth_pred / args.max_depth, 0, 1)
            
            # 套用 Inferno 顏色映射 (RGB)
            depth_vis = (cmap_color(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
            
            # 寫入影片幀
            out.write(depth_vis_bgr)
            
            # (選用) 即時預覽
            # cv2.imshow('MambaPixelFormer Video', depth_vis_bgr)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n[Done] Depth video saved to: {args.output_path}")

if __name__ == '__main__':
    main()