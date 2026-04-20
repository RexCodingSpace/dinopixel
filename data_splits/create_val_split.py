import random

# 讀取原始訓練檔案
with open("nyudepthv2_train_files_with_gt_dense.txt", "r") as f:
    lines = f.readlines()

print(f"Total training samples: {len(lines)}")

# 隨機打亂
random.seed(42)  # 固定種子，確保可重複
random.shuffle(lines)

# 切出 654 張作為 validation
val_lines = lines[:654]
train_lines = lines[654:]

print(f"New train samples: {len(train_lines)}")
print(f"Val samples: {len(val_lines)}")

# 儲存
with open("nyudepthv2_train_split_new.txt", "w") as f:
    f.writelines(train_lines)

with open("nyudepthv2_val_split_new.txt", "w") as f:
    f.writelines(val_lines)

print("Done! Created:")
print("  - pixelformer/data_splits/nyudepthv2_train_split.txt")
print("  - pixelformer/data_splits/nyudepthv2_val_split.txt")