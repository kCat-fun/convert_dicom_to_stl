import os
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def is_dicom_file(file_path):
    """
    ファイルがDICOMファイルかどうかを判定する
    """
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False

def normalize_pixel_array(pixel_array):
    """
    ピクセル配列を0-255の範囲に正規化する
    """
    # 最小値と最大値を取得
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)
    
    # 値の範囲が0の場合の処理
    if max_val == min_val:
        return np.zeros_like(pixel_array, dtype=np.uint8)
    
    # 0-255の範囲に正規化
    normalized = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def apply_window_level(pixel_array, window_center, window_width):
    """
    Window/Levelを適用してピクセル値を調整する
    """
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    
    # ウィンドウの範囲外の値をクリップ
    windowed = np.clip(pixel_array, min_val, max_val)
    
    # 0-255の範囲に正規化
    normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def convert_dicom_to_jpg(dicom_path, output_dir, quality=95):
    """
    単一のDICOMファイルをJPGに変換する
    
    Parameters:
    dicom_path: DICOMファイルのパス
    output_dir: 出力ディレクトリ
    quality: JPEG品質 (1-100)
    """
    try:
        # DICOMファイルを読み込み
        dicom_data = pydicom.dcmread(dicom_path)
        
        # ピクセルデータが存在するかチェック
        if not hasattr(dicom_data, 'pixel_array'):
            print(f"警告: {dicom_path} にはピクセルデータがありません。スキップします。")
            return False
        
        # ピクセル配列を取得
        pixel_array = dicom_data.pixel_array
        
        # カラー画像かグレースケール画像かを判定
        if len(pixel_array.shape) == 3:
            # カラー画像の場合
            if pixel_array.shape[2] == 3:  # RGB
                # 0-255の範囲に正規化
                if pixel_array.max() > 255:
                    pixel_array = normalize_pixel_array(pixel_array)
                image = Image.fromarray(pixel_array, mode='RGB')
            else:
                print(f"警告: {dicom_path} は対応していないカラー形式です。スキップします。")
                return False
        else:
            # グレースケール画像の場合
            # Window/Levelの情報があれば適用
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                window_center = dicom_data.WindowCenter
                window_width = dicom_data.WindowWidth
                
                # リストの場合は最初の値を使用
                if isinstance(window_center, (list, tuple)):
                    window_center = window_center[0]
                if isinstance(window_width, (list, tuple)):
                    window_width = window_width[0]
                
                pixel_array = apply_window_level(pixel_array, window_center, window_width)
            else:
                # Window/Level情報がない場合は単純に正規化
                pixel_array = normalize_pixel_array(pixel_array)
            
            # PILイメージに変換
            image = Image.fromarray(pixel_array, mode='L')
        
        # 出力ファイル名を生成
        input_filename = Path(dicom_path).stem
        output_filename = f"{input_filename}.jpg"
        output_path = Path(output_dir) / output_filename
        
        # JPGとして保存
        image.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        print(f"変換完了: {dicom_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"エラー: {dicom_path} の変換に失敗しました。{str(e)}")
        return False

def batch_convert_dicom_to_jpg(input_dir="/DICOM", output_dir="./converted_jpg", quality=95):
    """
    ディレクトリ内のすべてのDICOMファイルをJPGに変換する
    
    Parameters:
    input_dir: DICOMファイルが入っているディレクトリ
    output_dir: 変換後のJPGファイルを保存するディレクトリ
    quality: JPEG品質 (1-100)
    """
    
    # 入力ディレクトリの存在確認
    if not os.path.exists(input_dir):
        print(f"エラー: 入力ディレクトリ '{input_dir}' が存在しません。")
        return
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル一覧を取得
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    if not files:
        print(f"'{input_dir}' にファイルが見つかりません。")
        return
    
    # DICOMファイルの検出と変換
    dicom_files = []
    converted_count = 0
    error_count = 0
    
    print(f"ファイルをスキャンしています...")
    
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        
        # DICOMファイルかどうかを判定
        if is_dicom_file(file_path):
            dicom_files.append(file_path)
    
    print(f"{len(dicom_files)} 個のDICOMファイルが見つかりました。")
    
    # 各DICOMファイルを変換
    for i, dicom_path in enumerate(dicom_files, 1):
        print(f"変換中 ({i}/{len(dicom_files)}): {os.path.basename(dicom_path)}")
        
        if convert_dicom_to_jpg(dicom_path, output_dir, quality):
            converted_count += 1
        else:
            error_count += 1
    
    # 結果を表示
    print(f"\n=== 変換結果 ===")
    print(f"総ファイル数: {len(dicom_files)}")
    print(f"変換成功: {converted_count}")
    print(f"変換失敗: {error_count}")
    print(f"出力ディレクトリ: {output_dir}")

if __name__ == "__main__":
    # メイン処理
    input_directory = "./DICOM/00000000"  # DICOMファイルが入っているディレクトリ
    output_directory = "./output_jpg"  # 変換後のファイルを保存するディレクトリ
    jpeg_quality = 100  # JPEG品質 (1-100, 高いほど高品質)
    
    print("DICOM to JPG 変換プログラム")
    print("=" * 50)
    
    # バッチ変換実行
    batch_convert_dicom_to_jpg(input_directory, output_directory, jpeg_quality)