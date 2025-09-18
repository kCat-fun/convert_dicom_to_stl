import cv2
import os
import glob

input_dir = "output_jpg"
output_dir = "output_binarization"
os.makedirs(output_dir, exist_ok=True)

MIN_AREA = 70
extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.png", "*.bmp", "*.tif", "*.tiff"]

files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(input_dir, ext)))

for file_path in files:
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"読み込み失敗: {file_path}")
        continue

    otsu_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_val = min(otsu_thresh + 59, 255)
    _, binary = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            binary[labels == i] = 0

    # モルフォロジー膨張で線を太くし、ギャップを埋める
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_dilated = cv2.dilate(binary, kernel, iterations=2)

    # 輪郭検出＆塗りつぶし（膨張画像で）
    contours, _ = cv2.findContours(binary_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(binary_dilated, [cnt], -2, 255, thickness=cv2.FILLED)

    # 収縮で線を元に戻す（必要に応じて）
    binary_filled = cv2.erode(binary_dilated, kernel, iterations=2)

    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, binary_filled)
    print(f"保存完了: {output_path}")

print("すべての処理が完了しました。")
