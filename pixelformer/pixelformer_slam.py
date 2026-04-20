from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d
import threading
import time

from utils import post_process_depth, flip_lr
from networks.PixelFormer import PixelFormer

# NYUv2 標準內參
NYU_FX = 512.8579
NYU_FY = 512.8579
NYU_CX = 320.0
NYU_CY = 240.0


class RealtimeSLAM:
    """
    即時 SLAM + Open3D 動態視覺化
    """
    
    def __init__(self, fx, fy, cx, cy, max_depth=10.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.max_depth = max_depth
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 當前幀點雲（不累積）
        self.global_pcd = o3d.geometry.PointCloud()
        
        # 相機位姿
        self.current_pose = np.eye(4)
        self.trajectory = []
        
        # ORB
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_depth = None
        
        self.frame_count = 0
        self.keyframe_count = 0
        
        # 視覺化用
        self.camera_marker = None
        self.trajectory_line = None
        
    def create_camera_marker(self, size=0.3):
        """建立相機標記（視錐）- 指向 +Z 方向"""
        # 相機視錐，指向前方 (+Z)
        points = np.array([
            [0, 0, 0],                      # 相機中心
            [-size, -size*0.75, size*1.5],  # 左上
            [size, -size*0.75, size*1.5],   # 右上
            [size, size*0.75, size*1.5],    # 右下
            [-size, size*0.75, size*1.5],   # 左下
        ])
        
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # 從中心到四角
            [1, 2], [2, 3], [3, 4], [4, 1],  # 四角連線
        ]
        
        colors = [[0, 1, 0] for _ in lines]  # 綠色
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def update_camera_marker(self):
        """更新相機標記位置"""
        if self.camera_marker is None:
            self.camera_marker = self.create_camera_marker()
        
        # 基礎視錐頂點
        size = 0.3
        base_points = np.array([
            [0, 0, 0],
            [-size, -size*0.75, size*1.5],
            [size, -size*0.75, size*1.5],
            [size, size*0.75, size*1.5],
            [-size, size*0.75, size*1.5],
        ])
        
        # 變換到世界座標
        points_homo = np.hstack([base_points, np.ones((5, 1))])
        transformed = (self.current_pose @ points_homo.T).T[:, :3]
        
        self.camera_marker.points = o3d.utility.Vector3dVector(transformed)
        
        return self.camera_marker
    
    def update_trajectory_line(self):
        """更新軌跡線"""
        if len(self.trajectory) < 2:
            return None
        
        trajectory = np.array(self.trajectory)
        lines = [[i, i+1] for i in range(len(trajectory)-1)]
        
        if self.trajectory_line is None:
            self.trajectory_line = o3d.geometry.LineSet()
        
        self.trajectory_line.points = o3d.utility.Vector3dVector(trajectory)
        self.trajectory_line.lines = o3d.utility.Vector2iVector(lines)
        self.trajectory_line.colors = o3d.utility.Vector3dVector(
            [[1, 0, 0] for _ in lines]  # 紅色軌跡
        )
        
        return self.trajectory_line
    
    def depth_to_pointcloud(self, depth_map, rgb_image, subsample=2):
        """深度圖轉點雲 - 手動投影確保方向正確"""
        h, w = depth_map.shape
        
        # 子採樣 (subsample=2 表示每 2 個像素取 1 個)
        y_indices = np.arange(0, h, subsample)
        x_indices = np.arange(0, w, subsample)
        x_grid, y_grid = np.meshgrid(x_indices, y_indices)
        
        # 取深度和顏色
        z = depth_map[y_grid, x_grid]
        colors = rgb_image[y_grid, x_grid] / 255.0  # 歸一化到 0-1
        
        # 反投影到 3D（相機座標系）
        # X 向右, Y 向下, Z 向前（相機看的方向）
        x_cam = (x_grid - self.cx) * z / self.fx
        y_cam = (y_grid - self.cy) * z / self.fy
        z_cam = z  # 深度就是 Z
        
        # 整理成 Nx3
        points = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        
        # 過濾無效點
        mask = (points[:, 2] > 0.1) & (points[:, 2] < self.max_depth * 0.95)
        points = points[mask]
        colors = colors[mask]
        
        # 建立 Open3D 點雲
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 輕微下採樣避免太多重疊點
        if len(pcd.points) > 50000:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        return pcd
    
    def estimate_pose(self, gray, depth):
        """位姿估計"""
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if self.prev_gray is None or self.prev_des is None or len(kp) < 10:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            self.prev_depth = depth
            return True, np.eye(4)
        
        try:
            matches = self.bf.match(self.prev_des, des)
        except:
            return False, np.eye(4)
        
        if len(matches) < 10:
            return False, np.eye(4)
        
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(100, len(matches))]
        
        pts_prev = np.array([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.array([kp[m.trainIdx].pt for m in good_matches])
        
        pts_3d = []
        pts_2d = []
        
        h, w = self.prev_depth.shape
        for pt_prev, pt_curr in zip(pts_prev, pts_curr):
            x, y = int(pt_prev[0]), int(pt_prev[1])
            if 0 <= x < w and 0 <= y < h:
                z = self.prev_depth[y, x]
                if 0.1 < z < self.max_depth * 0.95:
                    X = (x - self.cx) * z / self.fx
                    Y = (y - self.cy) * z / self.fy
                    pts_3d.append([X, Y, z])
                    pts_2d.append(pt_curr)
        
        if len(pts_3d) < 6:
            return False, np.eye(4)
        
        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float64)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, None,
            iterationsCount=100,
            reprojectionError=8.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < 6:
            return False, np.eye(4)
        
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        self.prev_depth = depth
        
        return True, T
    
    def process_frame(self, rgb_image, depth_map, add_points=True):
        """處理單幀"""
        self.frame_count += 1
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        success, delta_T = self.estimate_pose(gray, depth_map)
        
        if success:
            try:
                self.current_pose = self.current_pose @ np.linalg.inv(delta_T)
            except:
                pass
        
        self.trajectory.append(self.current_pose[:3, 3].copy())
        
        # 產生當前幀點雲（不累積，每幀覆蓋）
        new_pcd = None
        if add_points:
            self.keyframe_count += 1
            pcd = self.depth_to_pointcloud(depth_map, rgb_image, subsample=2)
            
            if len(pcd.points) > 0:
                pcd.transform(self.current_pose)
                # 直接覆蓋，不累積
                self.global_pcd = pcd
                new_pcd = pcd
        
        return success, new_pcd


def run_realtime_slam(args):
    """即時 SLAM 主函數"""
    
    infer_h, infer_w = args.input_height, args.input_width
    
    # 載入模型
    print("Loading PixelFormer model...")
    model = PixelFormer(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    # 內參
    scale_x = infer_w / 640.0
    scale_y = infer_h / 480.0
    fx = NYU_FX * scale_x
    fy = NYU_FY * scale_y
    cx = NYU_CX * scale_x
    cy = NYU_CY * scale_y
    
    # SLAM
    slam = RealtimeSLAM(fx, fy, cx, cy, args.max_depth)
    
    # 影片
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {video_fps:.1f} FPS")
    
    # ========== Open3D 即時視覺化 ==========
    vis = o3d.visualization.Visualizer()
    vis.create_window("SLAM Live View", width=1280, height=720)
    
    # 初始化幾何物件
    pcd_vis = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_vis)
    
    camera_marker = slam.create_camera_marker()
    vis.add_geometry(camera_marker)
    
    trajectory_line = o3d.geometry.LineSet()
    vis.add_geometry(trajectory_line)
    
    # 渲染選項
    opt = vis.get_render_option()
    opt.point_size = 4.0  # 更大的點
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # 視角控制
    ctr = vis.get_view_control()
    
    # 初始視角：從後上方看向前方
    # 這樣可以看到相機和它前面的點雲
    first_view_set = False
    
    # ========== 主迴圈 ==========
    frame_idx = 0
    processed = 0
    
    print("\n=== SLAM Live View ===")
    print("Press Q in the 3D window to quit")
    print("========================\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                break
            
            frame_idx += 1
            if frame_idx % args.skip_frames != 0:
                continue
            
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            
            # 前處理
            frame_resized = cv2.resize(frame, (infer_w, infer_h))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            input_img = img_rgb.astype(np.float32)
            input_img[:, :, 0] = (input_img[:, :, 0] - 123.68) * 0.017
            input_img[:, :, 1] = (input_img[:, :, 1] - 116.78) * 0.017
            input_img[:, :, 2] = (input_img[:, :, 2] - 103.94) * 0.017
            
            input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).cuda()
            
            # 推論
            with torch.no_grad():
                depth = model(input_tensor)
                depth_flip = model(flip_lr(input_tensor))
                depth = post_process_depth(depth, depth_flip)
            
            depth_np = depth.cpu().squeeze().numpy()
            
            # SLAM 處理
            add_points = (processed % args.keyframe_interval == 0)
            success, _ = slam.process_frame(img_rgb, depth_np, add_points)
            
            # ===== 更新 3D 視覺化 =====
            
            # 更新點雲（每幀覆蓋，不累積）
            if len(slam.global_pcd.points) > 0:
                pcd_vis.points = slam.global_pcd.points
                pcd_vis.colors = slam.global_pcd.colors
                vis.update_geometry(pcd_vis)
            
            # 更新相機標記
            slam.update_camera_marker()
            camera_marker.points = slam.camera_marker.points
            camera_marker.lines = slam.camera_marker.lines
            camera_marker.colors = slam.camera_marker.colors
            vis.update_geometry(camera_marker)
            
            
            # 刷新畫面
            vis.poll_events()
            vis.update_renderer()
            
            # 第一幀或跟隨模式：設定視角從相機後方看
            if not first_view_set or args.follow_cam:
                try:
                    # 從相機位置往前看
                    cam_pos = slam.current_pose[:3, 3]
                    cam_forward = slam.current_pose[:3, 2]  # 相機的前方向量
                    
                    # 視角在相機後方，看向相機前方
                    ctr.set_lookat(cam_pos + cam_forward * 1)  # 看向前方
                    ctr.set_front(-cam_forward)  # 從後面看
                    ctr.set_up(-slam.current_pose[:3, 1])  # 相機的上方向
                    ctr.set_zoom(0.5)
                    first_view_set = True
                except:
                    pass
            
            # 顯示 2D 影像（可選）
            if args.show_2d:
                # 深度視覺化
                depth_vis = (depth_np / args.max_depth * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                
                status = "OK" if success else "LOST"
                pos = slam.current_pose[:3, 3]
                cv2.putText(frame_resized, f"Frame: {frame_idx} [{status}]", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_resized, f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame_resized, f"Points: {len(slam.global_pcd.points)}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                combined = np.hstack([frame_resized, depth_color])
                cv2.imshow('RGB + Depth', combined)
                
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    break
            
            processed += 1
            
            # 控制速度
            if args.delay > 0:
                time.sleep(args.delay / 1000.0)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    
    # 保存結果
    print(f"\nProcessed {processed} frames, {slam.keyframe_count} keyframes")
    
    # 保存最後一幀的點雲
    if len(slam.global_pcd.points) > 0:
        output_file = f"{args.output}_pointcloud.ply"
        o3d.io.write_point_cloud(output_file, slam.global_pcd)
        print(f"Saved: {output_file} ({len(slam.global_pcd.points)} points)")
    
    # 保存軌跡
    if len(slam.trajectory) > 0:
        np.savetxt(f"{args.output}_trajectory.txt", np.array(slam.trajectory), fmt='%.6f')
        print(f"Saved: {args.output}_trajectory.txt")
    
    # 保持視窗開啟讓使用者可以旋轉查看
    print("\n3D window still open. Close it to exit.")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='PixelFormer Realtime SLAM')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--max_depth', type=float, default=10.0)
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--input_width', type=int, default=640)
    parser.add_argument('--skip_frames', type=int, default=1, help='process every N frames')
    parser.add_argument('--max_frames', type=int, default=-1, help='max frames (-1 for all)')
    parser.add_argument('--keyframe_interval', type=int, default=1, help='add points every N processed frames (1=every frame)')
    parser.add_argument('--output', type=str, default='slam_live')
    parser.add_argument('--show_2d', action='store_true', help='also show 2D RGB+Depth window')
    parser.add_argument('--delay', type=int, default=0, help='delay between frames in ms')
    parser.add_argument('--follow_cam', action='store_true', help='camera view follows SLAM position')
    
    args = parser.parse_args()
    run_realtime_slam(args)


if __name__ == '__main__':
    main()