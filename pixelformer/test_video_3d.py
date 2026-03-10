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

# 確保你的環境可以 import 這些模組
from utils import post_process_depth, flip_lr
from networks.PixelFormer import PixelFormer

def main():
    parser = argparse.ArgumentParser(description='PixelFormer Video 3D Native Player')
    parser.add_argument('--model_name',      type=str,   default='pixelformer')
    parser.add_argument('--encoder',         type=str,   default='large07')
    parser.add_argument('--max_depth',       type=float, default=10)
    parser.add_argument('--checkpoint_path', type=str,   required=True, help='path to a checkpoint to load')
    parser.add_argument('--input_height',    type=int,   default=480)
    parser.add_argument('--input_width',     type=int,   default=640)
    parser.add_argument('--dataset',         type=str,   default='nyu')
    # 輸入與輸出設定
    parser.add_argument('--video',           type=str,   default='', required=True, help='path to input video')
    parser.add_argument('--output_video',    type=str,   default='output_pointcloud.mp4', help='path to output video')

    args = parser.parse_args()

    # 設定推論時的解析度
    infer_h, infer_w = args.input_height, args.input_width

    # ==========================================
    # NYUv2 標準內參 (保留您的設定)
    # ==========================================
    NYU_FX = 512.8579
    NYU_FY = 512.8579
    NYU_CX = 320.0
    NYU_CY = 240.0

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
    # 2. 開啟影片檔案
    # ==========================================
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    # 設定 VideoWriter (輸出 MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (infer_w, infer_h))

    print(f"Processing Video: {args.video}")
    print(f"Saving to: {args.output_video}")
    print("Press 'q' to stop early.")

    # ==========================================
    # 3. 核心：原視角點雲渲染 (Native Point Cloud)
    # ==========================================
    def render_native_view(img_rgb, depth_map):
        """
        將圖片變成「懸浮在黑色背景中」的點雲效果
        """
        # 1. 還原真實深度 Z (單位：公尺)
        z = args.max_depth * (1.0 - depth_map)
        
        # 2. 建立全黑畫布
        canvas = np.zeros((infer_h, infer_w, 3), dtype=np.uint8)

        # 3. 【關鍵修改】設定顯示過濾條件
        # z > 0.1 : 去掉貼在鏡頭上的極近雜訊
        # 這裡移除了 "z < max_depth * 0.95"，所以遠處的電視和廚房現在會顯示出來了！
        mask = (z > 0.1) 

        # 4. 把符合條件的像素填入畫布
        canvas[mask] = img_rgb[mask]

        return canvas

    # ==========================================
    # 4. 主迴圈 (逐幀處理)
    # ==========================================
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # 影片結束

            frame_count += 1

            # Resize 圖片
            frame_resized = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # --- Preprocessing ---
            input_img = img_rgb.astype(np.float32)
            input_img[:, :, 0] = (input_img[:, :, 0] - 123.68) * 0.017
            input_img[:, :, 1] = (input_img[:, :, 1] - 116.78) * 0.017
            input_img[:, :, 2] = (input_img[:, :, 2] - 103.94) * 0.017
            
            input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).cuda()

            # --- Inference ---
            with torch.no_grad():
                depth = model(input_tensor)
                # 若要追求極致速度，可註解掉下面兩行 Flip Test
                depth_flip = model(flip_lr(input_tensor))
                depth = post_process_depth(depth, depth_flip)
                
            depth_pred = depth.cpu().squeeze().numpy()

            # --- Visualization (渲染點雲) ---
            # 呼叫上面的渲染函式
            render_img = render_native_view(img_rgb, depth_pred)

            # 轉回 BGR 以便存檔
            render_bgr = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)

            # 寫入影片
            out.write(render_bgr)

            # 即時顯示
            cv2.imshow('Point Cloud Video', render_bgr)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("-" * 30)
        print(f"Done! Video saved to {args.output_video}")

if __name__ == '__main__':
    main()