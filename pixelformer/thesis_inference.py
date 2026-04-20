from __future__ import absolute_import, division, print_function
import torch
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

cmap_color  = plt.get_cmap('inferno')
cmap_error  = plt.get_cmap('coolwarm')

try:
    from utils import post_process_depth, flip_lr
    from networks.PixelFormer import MambaPixelFormer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')


def preprocess(frame_bgr, infer_h, infer_w):
    frame_resized = cv2.resize(frame_bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb[:, :, 0] = (img_rgb[:, :, 0] - 123.68) * 0.017
    img_rgb[:, :, 1] = (img_rgb[:, :, 1] - 116.78) * 0.017
    img_rgb[:, :, 2] = (img_rgb[:, :, 2] - 103.94) * 0.017
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).cuda()
    return tensor, frame_resized


def infer(model, tensor, no_flip_test):
    with torch.no_grad():
        depth = model(tensor)
        if not no_flip_test:
            depth_flip = model(flip_lr(tensor))
            depth = post_process_depth(depth, depth_flip)
    return depth.cpu().squeeze().numpy()


def depth_to_bgr(depth_np, max_depth):
    depth_norm = np.clip(depth_np / max_depth, 0, 1)
    depth_rgb = (cmap_color(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)


def eigen_crop(img, src_h, src_w):
    t = int(45  / 480 * src_h)
    b = int(471 / 480 * src_h)
    l = int(41  / 640 * src_w)
    r = int(601 / 640 * src_w)
    cropped = img[t:b, l:r]
    return cv2.resize(cropped, (src_w, src_h), interpolation=cv2.INTER_LINEAR)


def load_gt_depth_raw(gt_path, infer_h, infer_w):
    """回傳 float32 metres numpy，找不到回傳 None。"""
    if not os.path.isfile(gt_path):
        print(f"[Warning] GT depth not found: {gt_path}, skipping.")
        return None
    gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt_raw is None:
        print(f"[Warning] Cannot read GT depth: {gt_path}, skipping.")
        return None
    gt_m = gt_raw.astype(np.float32) / 1000.0
    gt_resized = cv2.resize(gt_m, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
    return eigen_crop(gt_resized, infer_h, infer_w)


def make_error_map_bgr(pred_np, gt_np, max_depth):
    valid = gt_np > 0
    err = np.zeros_like(pred_np)
    err[valid] = pred_np[valid] - gt_np[valid]
    err_norm = np.clip(err / max_depth * 0.5 + 0.5, 0, 1)
    err_rgb = (cmap_error(err_norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(err_rgb, cv2.COLOR_RGB2BGR)


def make_side_by_side(*imgs, border=3):
    h, w = imgs[0].shape[:2]
    n = len(imgs)
    total_w = w * n + border * (n + 1)
    total_h = h + border * 2
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    for i, img in enumerate(imgs):
        x = border + i * (w + border)
        canvas[border:border + h, x:x + w] = img
    return canvas


def main():
    parser = argparse.ArgumentParser(description='MambaPixelFormer Inference')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--max_depth',       type=float, default=10)
    parser.add_argument('--input_height',    type=int,   default=480)
    parser.add_argument('--input_width',     type=int,   default=640)
    parser.add_argument('--input',           type=str,   default='', help='圖片/影片路徑，留空用 Webcam')
    parser.add_argument('--output',          type=str,   default='', help='輸出目錄，留空存在當下路徑')
    parser.add_argument('--no_flip_test',    action='store_true', help='關閉 flip test 加快速度')
    args = parser.parse_args()

    infer_h, infer_w = args.input_height, args.input_width

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model = MambaPixelFormer(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    print("Model loaded.")

    is_image = args.input and args.input.lower().endswith(IMAGE_EXTS)

    if is_image:
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"Error: Cannot read image {args.input}")
            sys.exit(1)

        tensor, frame_resized = preprocess(frame, infer_h, infer_w)
        depth_np = infer(model, tensor, args.no_flip_test)

        # Eigen NYU Crop
        frame_cropped = eigen_crop(frame_resized, infer_h, infer_w)
        depth_cropped = eigen_crop(depth_np,      infer_h, infer_w)
        depth_bgr     = depth_to_bgr(depth_cropped, args.max_depth)

        # Sparse GT
        gt_path      = os.path.splitext(args.input)[0].replace('rgb_', 'sync_depth_') + '.png'
        gt_np        = load_gt_depth_raw(gt_path, infer_h, infer_w)
        gt_bgr       = depth_to_bgr(gt_np, args.max_depth) if gt_np is not None else None
        error_gt_bgr = make_error_map_bgr(depth_cropped, gt_np, args.max_depth) if gt_np is not None else None

        # Dense GT
        dense_gt_path = os.path.splitext(args.input)[0]\
            .replace('workspace/dataset/', 'workspace/dense_dataset/')\
            .replace('rgb_', 'sync_depth_') + '.png'
        dense_gt_np  = load_gt_depth_raw(dense_gt_path, infer_h, infer_w)
        dense_gt_bgr = depth_to_bgr(dense_gt_np, args.max_depth) if dense_gt_np is not None else None

        # 輸出路徑
        out_dir = args.output if args.output else '.'

        def save(name, img):
            if img is None:
                return
            p = os.path.join(out_dir, name)
            cv2.imwrite(p, img)
            print(f"Saved: {p}")

        save('result_rgb.png',       frame_cropped)
        save('result_depth.png',     depth_bgr)
        save('result_gt.png',        gt_bgr)
        save('result_error_map.png', error_gt_bgr)

        # Side-by-side 圖
        save('result_sbs_rgb_pred.png',
             make_side_by_side(frame_cropped, depth_bgr))

        if gt_bgr is not None:
            save('result_sbs_rgb_gt.png',
                 make_side_by_side(frame_cropped, gt_bgr))
            save('result_side_by_side_rgb_depth_gt.png',
                 make_side_by_side(frame_cropped, depth_bgr, gt_bgr))
            save('result_side_by_side_gt_depth.png',
                 make_side_by_side(gt_bgr, depth_bgr))

        if error_gt_bgr is not None:
            save('result_side_by_side_gt_error_map.png',
                 make_side_by_side(gt_bgr, error_gt_bgr))

    else:
        if args.input and os.path.isfile(args.input):
            print(f"Opening video: {args.input}")
            cap = cv2.VideoCapture(args.input)
        else:
            print("Opening Webcam (ID: 0)...")
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video source.")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0

        out_path = args.output if args.output else 'depth_side_by_side.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (infer_w * 2, infer_h))

        print(f"Output: {infer_w * 2}x{infer_h} @ {fps:.1f} fps → {out_path}")
        print("Press 'q' to stop.")

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                tensor, frame_resized = preprocess(frame, infer_h, infer_w)
                depth_np  = infer(model, tensor, args.no_flip_test)
                depth_bgr = depth_to_bgr(depth_np, args.max_depth)
                canvas    = make_side_by_side(frame_resized, depth_bgr)

                out.write(canvas)
                cv2.imshow('MambaPixelFormer: RGB + Depth', canvas)

                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user.")
                    break

        except KeyboardInterrupt:
            print("Interrupted.")
        except Exception as e:
            print(f"Error: {e}")
            raise
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"Done. {frame_count} frames → {out_path}")


if __name__ == '__main__':
    main()