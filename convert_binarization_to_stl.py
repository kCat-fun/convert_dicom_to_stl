import os
import glob
import cv2
import numpy as np
from skimage import measure
from stl import mesh
from scipy.ndimage import binary_fill_holes, binary_closing, generate_binary_structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ===== パラメータ =====
input_dir = "output_binarization"
output_stl = "bone_model.stl"
preview_png = "preview.png"

voxel_size = 4.0       # mm
resize_ratio = 0.9     # 少し縮小して軽量化
marching_step = 1      # 滑らかさ（1で最高）
close_iterations = 2   # 空洞除去の強さ

# ===== 大文字小文字両対応で画像取得 =====
extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

# ファイル名を数字順にソート
files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

if not files:
    raise RuntimeError("画像が見つかりません")

print(f"{len(files)} 枚の画像を読み込みます...")

# ===== 3D配列作成 =====
volume = []
for file_path in files:
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if resize_ratio < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * resize_ratio),
                               int(img.shape[0] * resize_ratio)),
                         interpolation=cv2.INTER_NEAREST)
    # 白を1、黒を0に変換
    volume.append((img == 255).astype(np.uint8))
volume = np.array(volume)

# ===== 3D空洞除去 =====
print("3D空洞を除去中...")
struct = generate_binary_structure(3, 3)  # 3D近傍
volume_filled = binary_closing(volume, structure=struct, iterations=close_iterations)
volume_filled = binary_fill_holes(volume_filled).astype(np.uint8)

# ===== Marching Cubes =====
print("Marching Cubesでメッシュ生成中...")
verts, faces, normals, values = measure.marching_cubes(
    volume_filled, level=0.5,
    spacing=(voxel_size, voxel_size, voxel_size),
    step_size=marching_step
)

# ===== STL保存 =====
print("STL保存中...")
solid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        solid_mesh.vectors[i][j] = verts[f[j], :]
solid_mesh.save(output_stl)
print(f"STLファイル保存完了: {output_stl}")

# ===== 軽量プレビュー画像生成 =====
print("プレビュー画像作成中...")
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
mesh_collection = Poly3DCollection(verts[faces], alpha=0.6)
mesh_collection.set_facecolor((0.7, 0.7, 0.8))
ax.add_collection3d(mesh_collection)

ax.set_xlim(0, volume_filled.shape[0] * voxel_size)
ax.set_ylim(0, volume_filled.shape[1] * voxel_size)
ax.set_zlim(0, volume_filled.shape[2] * voxel_size)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(preview_png, dpi=100)
plt.close()
print(f"プレビュー画像保存完了: {preview_png}")
