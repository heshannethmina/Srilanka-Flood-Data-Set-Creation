import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. SET YOUR FOLDER PATHS HERE (PLACEHOLDERS)
# ==========================================
SOURCE_DIR = Path("D:\DNN Research Project\Srilanka-Flood-multimodal-neural-network\kalu_ganga_pipeline\data\images")     # <-- Replace with source folder
OUTPUT_DIR = Path("D:\DNN Research Project\Srilanka-Flood-multimodal-neural-network\kalu_ganga_pipeline\data\png_images")     # <-- Replace with output folder


def convert_npy_to_png(src_folder: Path, dst_folder: Path):
    # Create the output directory if it doesn't exist
    dst_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files in the source directory
    npy_files = list(src_folder.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {src_folder.resolve()}")
        return

    print(f"Found {len(npy_files)} .npy files. Starting conversion...\n")
    
    successful = 0
    failed = 0

    for file_path in tqdm(npy_files, desc="Converting", unit="file"):
        try:
            # 1. Load numpy array
            data = np.load(file_path)
            
            # 2. Handle common dimension shapes (e.g., channels-first PyTorch tensors)
            data = np.squeeze(data)  # Remove single-dimensional entries (e.g., shape (1, H, W) -> (H, W))
            
            if data.ndim == 3 and data.shape[0] in [1, 3, 4]:
                # Transpose channels-first (C, H, W) -> channels-last (H, W, C)
                data = np.transpose(data, (1, 2, 0))
            
            # 3. Normalize pixel values to uint8 (0 to 255) if necessary
            if data.dtype != np.uint8:
                d_min, d_max = data.min(), data.max()
                if d_max > d_min:
                    data = ((data - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)
            
            # 4. Save as PNG
            img = Image.fromarray(data)
            output_file_path = dst_folder / f"{file_path.stem}.png"
            img.save(output_file_path)
            successful += 1

        except Exception as e:
            print(f"\nError converting {file_path.name}: {e}")
            failed += 1

    print(f"\nFinished! Converted {successful} files. Failed: {failed}.")

if __name__ == "__main__":
    convert_npy_to_png(SOURCE_DIR, OUTPUT_DIR)