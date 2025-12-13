"""
BISINDO Dataset Converter: YOLO Format → Cropped Images for CNN

📚 KONSEP DEEP LEARNING YANG DIPELAJARI:
===============================================

1. IMAGE PREPROCESSING (Persiapan Data)
   - Mengapa perlu preprocessing? Seperti cuci beras sebelum masak nasi.
   - Neural network butuh input yang konsisten dan standar.
   
2. NORMALISASI KOORDINAT (0-1 Range)
   - YOLO format: x,y,w,h dalam range 0.0 sampai 1.0 (normalized)
   - Kenapa? Supaya independent dari ukuran image (bisa 640x480 atau 1920x1080)
   - Analogi: Seperti persen (%), bukan nilai absolut
   
3. RESIZING KE 224x224
   - Kenapa 224? Standar dari ImageNet dataset (14 juta images)
   - Semua pretrained model (MobileNet, EfficientNet, ResNet) trained di size ini
   - Interpolation: Cara memperbesar/memperkecil image (bilinear, bicubic)
   - Trade-off: 224 cukup detail tapi tidak terlalu berat untuk GPU
   
4. RGB COLOR CHANNELS
   - 3 channels: Red, Green, Blue (0-255 per channel)
   - Shape tensor: (Height, Width, Channels) = (224, 224, 3)
   - Kenapa RGB, bukan grayscale? Warna kulit dan background penting untuk BISINDO
   
5. CLASS MAPPING (Label Encoding)
   - Folder A → class_id 0, B → 1, ..., Z → 25
   - Neural network butuh angka, bukan huruf
   - One-hot encoding nanti: [0,0,1,0,...,0] untuk class C (index 2)
   
6. DATA DISTRIBUTION ANALYSIS
   - Cek jumlah images per class (harus balanced)
   - Imbalanced data → model bias ke class yang banyak
   - Contoh: Class A=1000 images, Class Z=100 images → model lebih sering predict A

===============================================
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ============================================
# KONFIGURASI PATH
# ============================================
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "bisindo" / "images"
LABELS_DIR = BASE_DIR / "bisindo" / "labels"
OUTPUT_DIR = BASE_DIR / "dataset" / "cropped"
STATS_FILE = BASE_DIR / "dataset" / "data_stats.json"
ERROR_LOG = BASE_DIR / "dataset" / "conversion_errors.log"

# Target size untuk CNN input (standar ImageNet)
TARGET_SIZE = (224, 224)

# Mapping folder name → class_id (A=0, B=1, ..., Z=25)
LETTER_TO_CLASS = {chr(65 + i): i for i in range(26)}  # A-Z


# ============================================
# FUNGSI UTILITY
# ============================================

def parse_yolo_label(label_path):
    """
    Parse YOLO format label file.
    
    Format: class_id x_center y_center width height
    Semua nilai dalam range 0.0 - 1.0 (normalized)
    
    Returns:
        tuple: (class_id, x_center, y_center, width, height)
    """
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                return None
            
            parts = line.split()
            if len(parts) != 5:
                return None
            
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            
            # Validasi range (harus 0-1)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                return None
                
            return class_id, x, y, w, h
            
    except Exception as e:
        return None


def yolo_to_bbox(img_shape, x_center, y_center, width, height):
    """
    Konversi YOLO normalized coordinates → pixel coordinates.
    
    📚 KONSEP: Coordinate Transformation
    - YOLO: (x_center, y_center, width, height) dalam 0-1
    - OpenCV: (x1, y1, x2, y2) dalam pixels
    
    Contoh:
    Image 640x480, YOLO (0.5, 0.5, 0.8, 0.6)
    → Center: (320, 240), Size: (512, 288)
    → BBox: x1=64, y1=96, x2=576, y2=384
    """
    img_h, img_w = img_shape[:2]
    
    # Convert normalized → pixel
    x_center_px = x_center * img_w
    y_center_px = y_center * img_h
    width_px = width * img_w
    height_px = height * img_h
    
    # Convert center+size → top-left + bottom-right
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    
    return x1, y1, x2, y2


def crop_and_resize(image, bbox, target_size=TARGET_SIZE):
    """
    Crop bounding box dari image dan resize ke target size.
    
    📚 KONSEP: Resizing & Interpolation
    - Interpolation methods:
      * INTER_LINEAR (bilinear): Cepat, kualitas bagus
      * INTER_CUBIC (bicubic): Lebih halus, sedikit lebih lambat
      * INTER_AREA: Terbaik untuk downscaling
    
    - Aspect ratio: Kita paksa jadi square (224x224) karena:
      * CNN expect fixed input size
      * Hand gestures biasanya tidak depend on aspect ratio
    """
    x1, y1, x2, y2 = bbox
    
    # Crop region
    cropped = image[y1:y2, x1:x2]
    
    # Check if crop is valid (min 10x10 pixels)
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return None
    
    # Resize dengan interpolation (bilinear untuk balance speed & quality)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized


def log_error(error_log_file, message):
    """Log error ke file untuk review nanti."""
    with open(error_log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")


# ============================================
# MAIN CONVERSION FUNCTION
# ============================================

def convert_dataset():
    """
    Main function untuk konversi dataset YOLO → cropped images.
    
    Process:
    1. Iterate semua splits (train, val)
    2. Iterate semua letter folders (A-Z)
    3. Load image + label
    4. Parse YOLO bbox
    5. Crop & resize
    6. Save dengan class_id yang benar
    7. Collect statistics
    """
    
    print("=" * 60)
    print("BISINDO DATASET CONVERSION: YOLO → CNN Format")
    print("=" * 60)
    print()
    print("📚 Apa yang terjadi dalam proses ini?")
    print("   1. Membaca 11,470 images dengan bounding box annotations")
    print("   2. Crop region tangan dari setiap image")
    print("   3. Resize ke 224x224 pixels (standar ImageNet)")
    print("   4. Organize per class (A-Z) untuk training")
    print()
    print("💡 Kenapa perlu cropping?")
    print("   - Object detection (YOLO): Deteksi DI MANA tangan berada")
    print("   - Classification (CNN): Tangan ADALAH APA (huruf A-Z)")
    print("   - Kita sudah tahu lokasinya, tinggal klasifikasi!")
    print()
    print("=" * 60)
    print()
    
    # Prepare output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / "train", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "val", exist_ok=True)
    
    # Clear error log
    if ERROR_LOG.exists():
        ERROR_LOG.unlink()
    
    # Statistics
    stats = {
        'total_processed': 0,
        'total_success': 0,
        'total_errors': 0,
        'splits': {}
    }
    
    # Process each split (train, val)
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing: {split.upper()} SET")
        print(f"{'='*60}\n")
        
        split_images_dir = IMAGES_DIR / split
        split_labels_dir = LABELS_DIR / split
        split_output_dir = OUTPUT_DIR / split
        
        split_stats = defaultdict(int)
        
        # Process each letter folder (A-Z)
        for letter in sorted(os.listdir(split_images_dir)):
            letter_path = split_images_dir / letter
            
            if not letter_path.is_dir():
                continue
            
            # Validate letter name (harus A-Z)
            if letter not in LETTER_TO_CLASS:
                print(f"⚠️  Skipping invalid folder: {letter}")
                continue
            
            class_id = LETTER_TO_CLASS[letter]
            
            # Create output folder untuk class ini
            output_class_dir = split_output_dir / letter
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all images in this folder
            image_files = list(letter_path.glob("*.jpg")) + list(letter_path.glob("*.png"))
            
            print(f"📁 {letter} (class_id={class_id}): {len(image_files)} images")
            
            success_count = 0
            error_count = 0
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"  Converting {letter}", ncols=80):
                stats['total_processed'] += 1
                
                # Find corresponding label file
                label_path = split_labels_dir / letter / f"{img_path.stem}.txt"
                
                # Check if label exists
                if not label_path.exists():
                    error_count += 1
                    log_error(ERROR_LOG, f"Missing label: {img_path} → {label_path}")
                    continue
                
                # Parse YOLO label
                parsed = parse_yolo_label(label_path)
                if parsed is None:
                    error_count += 1
                    log_error(ERROR_LOG, f"Invalid label format: {label_path}")
                    continue
                
                _, x, y, w, h = parsed
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    error_count += 1
                    log_error(ERROR_LOG, f"Cannot load image: {img_path}")
                    continue
                
                # Convert YOLO → bbox
                bbox = yolo_to_bbox(image.shape, x, y, w, h)
                
                # Crop and resize
                cropped = crop_and_resize(image, bbox, TARGET_SIZE)
                if cropped is None:
                    error_count += 1
                    log_error(ERROR_LOG, f"Invalid crop (too small): {img_path}")
                    continue
                
                # Save cropped image
                output_path = output_class_dir / img_path.name
                cv2.imwrite(str(output_path), cropped)
                
                success_count += 1
                stats['total_success'] += 1
            
            # Update statistics
            split_stats[letter] = {
                'success': success_count,
                'errors': error_count,
                'total': len(image_files)
            }
            stats['total_errors'] += error_count
            
            print(f"   ✅ Success: {success_count} | ❌ Errors: {error_count}")
        
        stats['splits'][split] = dict(split_stats)
    
    # Save statistics
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}\n")
    
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Total Processed: {stats['total_processed']}")
    print(f"✅ Total Success:   {stats['total_success']}")
    print(f"❌ Total Errors:    {stats['total_errors']}")
    print(f"📈 Success Rate:    {stats['total_success']/stats['total_processed']*100:.2f}%")
    print()
    print(f"💾 Statistics saved to: {STATS_FILE}")
    if stats['total_errors'] > 0:
        print(f"📝 Error log saved to: {ERROR_LOG}")
    print()
    
    # Display per-class distribution
    print("📊 CLASS DISTRIBUTION (Train Set):")
    print("-" * 60)
    print(f"{'Letter':<8} {'Success':<10} {'Errors':<10} {'Total':<10}")
    print("-" * 60)
    
    train_stats = stats['splits']['train']
    for letter in sorted(train_stats.keys()):
        s = train_stats[letter]
        print(f"{letter:<8} {s['success']:<10} {s['errors']:<10} {s['total']:<10}")
    
    print("-" * 60)
    print()
    
    # Check for imbalanced data
    print("🔍 CHECKING DATA BALANCE...")
    train_counts = [train_stats[l]['success'] for l in sorted(train_stats.keys())]
    min_count = min(train_counts)
    max_count = max(train_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    
    print(f"   Min images per class: {min_count}")
    print(f"   Max images per class: {max_count}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2.0:
        print("   ⚠️  WARNING: Dataset cukup imbalanced!")
        print("      → Model mungkin bias ke class dengan data banyak")
        print("      → Consider: class weights atau resampling saat training")
    else:
        print("   ✅ Dataset cukup balanced!")
    
    print()
    print("=" * 60)
    print("CONVERSION COMPLETE! 🎉")
    print("=" * 60)
    print()
    print("📚 NEXT STEPS:")
    print("   1. Review data_stats.json untuk validasi distribusi")
    print("   2. Check conversion_errors.log jika ada errors")
    print("   3. Lanjut ke training script: train_mobilenetv3.py")
    print()
    print("💡 KONSEP PENTING YANG SUDAH DIPELAJARI:")
    print("   ✓ Image preprocessing & normalization")
    print("   ✓ Coordinate transformation (YOLO → pixel)")
    print("   ✓ Resizing & interpolation methods")
    print("   ✓ RGB color channels (3D tensor)")
    print("   ✓ Class mapping & label encoding")
    print("   ✓ Data distribution & balance analysis")
    print()


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        convert_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion interrupted by user!")
    except Exception as e:
        print(f"\n\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
