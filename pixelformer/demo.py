from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import time
import argparse
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

# 定義 colormap
greys = plt.get_cmap('Greys')

from utils import post_process_depth, flip_lr
from networks.PixelFormer import MambaPixelFormer

# Argument Parser
parser = argparse.ArgumentParser(description='PixelFormer Headless Demo')
parser.add_argument('--model_name',      type=str,   help='model name', default='pixelformer')
parser.add_argument('--encoder',         type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--max_depth',       type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str,   help='path to a checkpoint to load', required=True)
parser.add_argument('--input_height',    type=int,   help='input height', default=480)
parser.add_argument('--input_width',     type=int,   help='input width',  default=640)
parser.add_argument('--dataset',         type=str,   help='dataset this model trained on',  default='nyu')
parser.add_argument('--crop',            type=str,   help='crop: kbcrop, edge, non',  default='non')
parser.add_argument('--video',           type=str,   help='video path',  default='')

args = parser.parse_args()

# Image shapes
height_rgb, width_rgb = args.input_height, args.input_width
height_depth, width_depth = height_rgb, width_rgb

# =============== Intrinsics rectify (Optional) ==================
Use_intrs_remap = False
camera_matrix = np.zeros(shape=(3, 3))
camera_matrix[0, 0] = 5.4765313594010649e+02
camera_matrix[0, 2] = 3.2516069906172453e+02
camera_matrix[1, 1] = 5.4801781476172562e+02
camera_matrix[1, 2] = 2.4794113960783835e+02
camera_matrix[2, 2] = 1
dist_coeffs = np.array([ 3.7230261423972011e-02, -1.6171708069773008e-01, -3.5260752900266357e-04, 1.7161234226767313e-04, 1.0192711400840315e-01 ])
new_camera_matrix = np.zeros(shape=(3, 3))
new_camera_matrix[0, 0] = 518.8579
new_camera_matrix[0, 2] = 320
new_camera_matrix[1, 1] = 518.8579
new_camera_matrix[1, 2] = 240
new_camera_matrix[2, 2] = 1

R = np.identity(3, dtype=float) # Fixed np.float error
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R, new_camera_matrix, (width_rgb, height_rgb), cv2.CV_32FC1)

def load_model():
    args.mode = 'test'
    model = MambaPixelFormer(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    return model

def get_input_frame():
    """自動判斷輸入是影片還是圖片，並讀取"""
    path = args.video
    
    if path:
        print(f"Loading input from: {path}")
        
        # 1. 檢查是否為圖片格式 (根據副檔名)
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            frame = cv2.imread(path)
            if frame is not None:
                print("Successfully read image file.")
                return frame
            else:
                print("Error: Failed to read image via cv2.imread.")
        
        # 2. 如果不是圖片，就嘗試當作影片讀取
        else:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("Successfully captured frame from video.")
                return frame
            else:
                print("Failed to read video.")

    # 3. 如果沒給路徑，或讀取失敗，嘗試讀取 Webcam
    print("Attempting to capture from Webcam (ID: 0)...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("Successfully captured from webcam.")
            return frame
    
    # 4. 全部失敗，生成雜訊圖 (避免程式當掉)
    print("Warning: No valid input found. Generating random noise image.")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame

def save_point_cloud(filename, rgb_image, depth_map, max_dist=10.0):
    """
    將 RGB 和 Depth 轉換為 .ply 點雲檔案
    rgb_image: (H, W, 3) uint8
    depth_map: (H, W) float32 (Normalized 0~1)
    max_dist: 真實世界的最大距離 (meters)
    """
    print(f"Generating Point Cloud: {filename}...")
    
    # 1. 還原真實深度 (Meters)
    z = depth_map * max_dist
    
    # 2. 產生相機參數 (如果沒有內參，就用寬度當焦距的通用假設)
    h, w = z.shape
    fx = w * 1.0  # 焦距 x
    fy = w * 1.0  # 焦距 y
    cx = w / 2.0  # 光心 x
    cy = h / 2.0  # 光心 y
    
    # 3. 產生網格座標 (u, v)
    x_row = np.arange(w)
    y_col = np.arange(h)
    x_grid, y_grid = np.meshgrid(x_row, y_col)
    
    # 4. 反投影公式：從 Pixel (u, v, d) 轉成 World (x, y, z)
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    x_world = (x_grid - cx) * z / fx
    y_world = (y_grid - cy) * z / fy
    
    # 5. 整理資料
    points = np.stack([x_world, y_world, z], axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3)
    
    # 6. 過濾掉無效點 (太遠或深度為0)
    mask = (points[:, 2] > 0) & (points[:, 2] < max_dist * 0.95)
    points = points[mask]
    colors = colors[mask]
    
    # 7. 寫入 PLY 檔案 (ASCII 格式，任何 3D 軟體如 MeshLab 都可開)
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
            r, g, b = colors[i]  # OpenCV讀進來是BGR，這裡要注意
            # 如果輸入是 RGB，直接寫；如果是 BGR，要轉成 r, g, b
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            
    print("Point Cloud saved successfully.")

def run_inference_headless():
    # 1. Load Model
    model = load_model()
    print("Model loaded successfully.")

    # 2. Get Input
    frame = get_input_frame()
    
    # Preprocess (Resize / Undistort)
    if Use_intrs_remap:
        frame_ud = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    else:
        frame_ud = cv2.resize(frame, (width_rgb, height_rgb), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2RGB)
    
    # 3. Prepare Tensor
    input_image = img_rgb.astype(np.float32)
    # Normalize
    input_image[:, :, 0] = (input_image[:, :, 0] - 123.68) * 0.017
    input_image[:, :, 1] = (input_image[:, :, 1] - 116.78) * 0.017
    input_image[:, :, 2] = (input_image[:, :, 2] - 103.94) * 0.017

    H, W, _ = input_image.shape
    
    # Crop Logic
    if args.crop == 'kbcrop':
        top_margin = int(H - 352)
        left_margin = int((W - 1216) / 2)
        input_image_cropped = input_image[top_margin:top_margin + 352, left_margin:left_margin + 1216]
    elif args.crop == 'edge':
        input_image_cropped = input_image[32:-32, 32:-32, :]
    else:
        input_image_cropped = input_image

    input_images = np.expand_dims(input_image_cropped, axis=0)
    input_images = np.transpose(input_images, (0, 3, 1, 2))

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        image_var = Variable(torch.from_numpy(input_images)).cuda()
        depth_est = model(image_var)
        
        # Post Processing
        image_flipped = flip_lr(image_var)
        depth_est_flipped = model(image_flipped)
        depth_cropped = post_process_depth(depth_est, depth_est_flipped)

    # 5. Reconstruct Depth Map
    depth = np.zeros((height_depth, width_depth), dtype=np.float32)
    if args.crop == 'kbcrop':
        depth[top_margin:top_margin + 352, left_margin:left_margin + 1216] = depth_cropped[0].cpu().squeeze() / args.max_depth
    elif args.crop == 'edge':
        depth[32:-32, 32:-32] = depth_cropped[0].cpu().squeeze() / args.max_depth
    else:
        depth[:, :] = depth_cropped[0].cpu().squeeze() / args.max_depth

    # 6. Colorize & Save
    # Apply Greys colormap (Output is RGB)
    colored_depth = (greys(np.log10(depth * args.max_depth))[:, :, :3] * 255).astype('uint8')
    colored_depth_bgr = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)

    # Save files
    cv2.imwrite("result_rgb.png", frame_ud)       # 這是 BGR
    cv2.imwrite("result_depth.png", colored_depth_bgr)
    
    # =========== 新增這行：輸出 3D 點雲 ===========
    # 注意：save_point_cloud 需要 RGB 格式，而 frame_ud 是 BGR，所以要轉一下
    frame_rgb = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2RGB)
    save_point_cloud("result_3d.ply", frame_rgb, depth, args.max_depth)
    # ============================================

    print("-" * 30)
    print("Done! Results saved to:")
    print("  -> result_rgb.png")
    print("  -> result_depth.png")
    print("  -> result_3d.ply  <-- 用 MeshLab 或 Open3D 打開這個！")
    print("-" * 30)

if __name__ == '__main__':
    run_inference_headless()