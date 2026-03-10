from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定義 colormap
greys = plt.get_cmap('Greys')
# 建議：可以用 inferno 或 magma 讓深度影片更清楚，若要維持黑白請保留 Greys
cmap_color = plt.get_cmap('inferno') 

# 確保你的環境可以 import 這些模組
from utils import post_process_depth, flip_lr
from networks.PixelFormer import PixelFormer

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='PixelFormer Video Inference')
    parser.add_argument('--model_name',      type=str,   default='pixelformer')
    parser.add_argument('--encoder',         type=str,   default='large07')
    parser.add_argument('--max_depth',       type=float, default=10)
    parser.add_argument('--checkpoint_path', type=str,   required=True, help='path to a checkpoint to load')
    parser.add_argument('--input_height',    type=int,   default=480)
    parser.add_argument('--input_width',     type=int,   default=640)
    parser.add_argument('--dataset',         type=str,   default='nyu')
    parser.add_argument('--video',           type=str,   default='', help='path to input video')
    parser.add_argument('--output_video',    type=str,   default='depth_output.mp4', help='path to output video')

    args = parser.parse_args()

    # 設定推論時的解析度
    infer_h, infer_w = args.input_height, args.input_width

    # ==========================================
    # 1. 載入模型
    # ==========================================
    def load_model():
        print(f"Loading checkpoint: {args.checkpoint_path}")
        if not os.path.exists(args.checkpoint_path):
            print(f"Error: Checkpoint not found at {args.checkpoint_path}")
            sys.exit(1)
            
        model = PixelFormer(version='large07', inv_depth=False, max_depth=args.max_depth)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.cuda()
        return model

    model = load_model()

    # ==========================================
    # 2. 設定影片輸入與輸出
    # ==========================================
    if args.video and os.path.isfile(args.video):
        print(f"Opening video: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print("Opening Webcam (ID: 0)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        sys.exit(1)

    # 取得原始影片 FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # 設定 VideoWriter (輸出 MP4)
    # 注意：輸出尺寸設為與推論尺寸相同 (infer_w, infer_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (infer_w, infer_h))

    print(f"Processing... Press 'q' to stop.")
    print(f"Output will be saved to: {args.output_video}")

    # ==========================================
    # 3. 逐幀處理迴圈
    # ==========================================
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # 影片結束

            frame_count += 1
            
            # Resize 到模型需要的大小
            frame_resized = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # --- 資料前處理 (Preprocessing) ---
            input_img = img_rgb.astype(np.float32)
            input_img[:, :, 0] = (input_img[:, :, 0] - 123.68) * 0.017
            input_img[:, :, 1] = (input_img[:, :, 1] - 116.78) * 0.017
            input_img[:, :, 2] = (input_img[:, :, 2] - 103.94) * 0.017
            
            input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).cuda()

            # --- 模型推論 (Inference) ---
            with torch.no_grad():
                depth = model(input_tensor)
                
                # Flip Test: 雖然會增加準確度，但會讓速度變慢一倍
                # 如果要即時性，可以註解掉下面兩行，直接用 depth
                depth_flip = model(flip_lr(input_tensor))
                depth = post_process_depth(depth, depth_flip)

            # --- 後處理 (Visualization) ---
            depth_pred = depth.cpu().squeeze().numpy() / args.max_depth
            
            # 轉成視覺化圖片 (使用 Log 讓層次更明顯)
            # 這裡改用彩色 colormap (如 inferno)，若堅持要黑白改回 greys 即可
            depth_vis = (cmap_color(depth_pred)[:, :, :3] * 255).astype(np.uint8)
            
            # Matplotlib 輸出是 RGB，OpenCV 存檔需要 BGR
            depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

            # 寫入影片
            out.write(depth_vis_bgr)

            # 顯示處理進度
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
            
            # (選用) 即時預覽視窗
            cv2.imshow('Depth Video', depth_vis_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # 釋放資源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("-" * 30)
        print(f"Done! Video saved as {args.output_video}")

if __name__ == '__main__':
    main()