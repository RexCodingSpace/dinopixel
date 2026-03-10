import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
# 讀取你剛剛生成的 .ply 檔案
pcd = o3d.io.read_point_cloud("result_3d.ply")

# 印出這張圖有多少個點
print(pcd)
print(np.asarray(pcd.points))

# 開啟視窗顯示 (可以用滑鼠左鍵旋轉，右鍵拖曳，滾輪縮放)
o3d.visualization.draw_geometries([pcd], 
                                  window_name="PixelFormer 3D Result",
                                  width=800,
                                  height=600)