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
import open3d as o3d  # 新增: 引入 Open3D

# 定義 colormap
greys = plt.get_cmap('Greys')
cmap_color = plt.get_cmap('inferno') 

# 確保你的環境可以 import 這些模組 (請確認 utils 和 networks 資料夾在同一目錄下)
from utils import post_process_depth, flip_lr
from networks.PixelFormer import PixelFormer

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='PixelFormer Simple 3D Demo (Fixed NYU Intrinsics)')
    parser.add_argument('--model_name',      type=str,   default='pixelformer')
    parser.add_argument('--encoder',         type=str,   default='large07')
    parser.add_argument('--max_depth',       type=float, default=10)
    parser.add_argument('--checkpoint_path', type=str,   required=True, help='path to a checkpoint to load')
    parser.add_argument('--input_height',    type=int,   default=480)
    parser.add_argument('--input_width',     type=int,   default=640)
    parser.add_argument('--dataset',         type=str,   default='nyu')
    parser.add_argument('--video',           type=str,   default='', help='path to image or video')
    parser.add_argument('--no_vis',          action='store_true', help='If set, disable Open3D visualization') # 新增: 可選參數關閉視窗

    args = parser.parse_args()

    # 設定推論時的解析度 (通常是 640x480)
    infer_h, infer_w = args.input_height, args.input_width

    # ==========================================
    # 核心設定：NYUv2 標準內參 (Hardcoded)
    # ==========================================
    NYU_FX = 512.8579
    NYU_FY = 512.8579
    NYU_CX = 320.0
    NYU_CY = 240.0

    def load_model():
        """載入 PixelFormer 模型"""
        args.mode = 'test'
        # 注意: 請確保 PixelFormer 類別定義與權重檔匹配
        model = PixelFormer(version='large07', inv_depth=False, max_depth=args.max_depth)
        model = torch.nn.DataParallel(model)

        print(f"Loading checkpoint: {args.checkpoint_path}")
        if not os.path.exists(args.checkpoint_path):
            print(f"Error: Checkpoint not found at {args.checkpoint_path}")
            sys.exit(1)
            
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.cuda()
        return model

    def get_input_frame():
        """讀取輸入 (圖片/影片/Webcam)"""
        path = args.video
        if path:
            print(f"Loading input from: {path}")
            if os.path.isfile(path):
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    frame = cv2.imread(path)
                    if frame is not None: return frame
                else:
                    cap = cv2.VideoCapture(path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret: return frame
            else:
                print(f"Error: File not found: {path}")
                sys.exit(1)
        
        print("Attempting to capture from Webcam (ID: 0)...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret: return frame
        
        print("Warning: No input source. Generating noise.")
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def save_point_cloud(filename, rgb_image, depth_map, max_dist=10.0):
        """生成 3D 點雲 (.ply)"""
        print(f"Generating Point Cloud: {filename}...")
        
        h, w = depth_map.shape
        z = max_dist * (1.0 - depth_map) 
        
        scale_x = w / 640.0
        scale_y = h / 480.0
        
        fx = NYU_FX * scale_x
        fy = NYU_FY * scale_y
        cx = NYU_CX * scale_x
        cy = NYU_CY * scale_y
        
        x_row = np.arange(w)
        y_col = np.arange(h)
        x_grid, y_grid = np.meshgrid(x_row, y_col)
        
        x_world = (x_grid - cx) * z / fx
        y_world = -(y_grid - cy) * z / fy
        
        points = np.stack([x_world, y_world, z], axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) 
        
        mask = (points[:, 2] > 0.1) & (points[:, 2] < max_dist * 0.95)
        points = points[mask]
        colors = colors[mask]
        
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(filename, 'w') as f:
            f.write(header)
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
                
        print(f"Saved {filename} ({len(points)} points).")

    # =====================
    # 主執行流程
    # =====================
    
    # 1. 載入模型
    model = load_model()

    # 2. 取得圖片
    frame = get_input_frame()
    
    # 3. 縮放與前處理
    frame_resized = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    input_img = img_rgb.astype(np.float32)
    input_img[:, :, 0] = (input_img[:, :, 0] - 123.68) * 0.017
    input_img[:, :, 1] = (input_img[:, :, 1] - 116.78) * 0.017
    input_img[:, :, 2] = (input_img[:, :, 2] - 103.94) * 0.017
    
    input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).cuda()
    
    # 4. 推論
    print("Running inference...")
    with torch.no_grad():
        depth = model(input_tensor)
        depth_flip = model(flip_lr(input_tensor))
        depth = post_process_depth(depth, depth_flip)
        
    depth_pred = depth.cpu().squeeze().numpy() / args.max_depth

    # 5. 存檔結果
    depth_vis = (cmap_color(np.log10(depth_pred * args.max_depth))[:, :, :3] * 255).astype(np.uint8)
    depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("result_rgb.png", frame_resized)
    cv2.imwrite("result_depth.png", depth_vis_bgr)
    
    ply_filename = "result_3d.ply"
    save_point_cloud(ply_filename, img_rgb, depth_pred, args.max_depth)
    
    print("-" * 30)
    print("Processing Done.")

    # =====================
    # 6. Open3D 視覺化整合
    # =====================
    if not args.no_vis:
        print("\n[Open3D] Loading point cloud for visualization...")
        try:
            # 讀取剛剛生成的 .ply 檔案
            pcd = o3d.io.read_point_cloud(ply_filename)

            # 印出資訊
            print(f"[Open3D] Point Cloud loaded: {pcd}")
            # 如果想看點座標數值，可以取消下面這行的註解
            # print(np.asarray(pcd.points))

            print("[Open3D] Opening window... (Close window to exit script)")
            # 開啟視窗顯示
            o3d.visualization.draw_geometries([pcd], 
                                              window_name="PixelFormer 3D Result",
                                              width=800,
                                              height=600)
        except Exception as e:
            print(f"[Open3D Error] Visualization failed: {e}")
    else:
        print("Visualization skipped (--no_vis specified).")

if __name__ == '__main__':
    main()