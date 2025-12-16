"""
MediaPipe Landmarks Extraction for Multi-Modal Learning

📚 KONSEP MULTI-MODAL LEARNING:
===============================================

18. MULTI-MODAL LEARNING
    Multi-modal = belajar dari multiple types of data (modalities)

    Contoh Multi-Modal Systems:
    - Video understanding: Image frames + Audio + Subtitles
    - Medical diagnosis: X-ray + Blood test + Patient history
    - Sign language: Hand images + Hand landmarks + Body pose

    Kenapa Multi-Modal?
    - Redundancy: Jika satu modality gagal, lainnya backup
    - Complementarity: Setiap modality capture informasi berbeda
    - Robustness: Lebih tahan terhadap noise/errors

19. MEDIAPIPE HAND LANDMARKS
    MediaPipe detect 21 key points di tangan:
    - 0: Wrist (pergelangan)
    - 1-4: Thumb (jempol) - dari base ke tip
    - 5-8: Index finger (telunjuk)
    - 9-12: Middle finger (jari tengah)
    - 13-16: Ring finger (jari manis)
    - 17-20: Pinky (kelingking)

    Setiap landmark punya 3 coordinates:
    - x: horizontal position (0-1, normalized ke image width)
    - y: vertical position (0-1, normalized ke image height)
    - z: depth relative to wrist (real-world scale in cm)

20. LANDMARK NORMALIZATION
    Problem: Landmarks berubah dengan posisi/scale tangan
    Solution: Normalize relative to wrist (landmark 0)

    Normalized coordinates:
    - x' = (x - wrist_x) / bbox_width
    - y' = (y - wrist_y) / bbox_height
    - z' = z - wrist_z

    Result: Invariant terhadap translation & scale!

21. FEATURE ENGINEERING DARI LANDMARKS
    🔥 2-HAND SUPPORT:
    - Single hand: 21 × 3 = 63 features
    - Two hands: 63 × 2 = 126 features

    Untuk dataset BISINDO yang memiliki banyak huruf 2-tangan
    (seperti C, G, H, J, dll), kita ekstraksi hingga 2 tangan.

    Handling:
    - 2 tangan terdeteksi → 126 features (tangan 1 + tangan 2)
    - 1 tangan terdeteksi → 126 features (tangan 1 + 63 zeros)
    - 0 tangan → None (skip image ini)

    Bisa tambah engineered features:
    - Angles: Sudut antar jari (e.g., thumb-index angle)
    - Distances: Jarak relatif (e.g., thumb tip to palm)
    - Ratios: Proporsi jari (e.g., index/middle length ratio)
    - Inter-hand: Jarak/angle antar kedua tangan (untuk 2-hand gestures)

    Trade-off:
    - More features = more expressive
    - Too many features = overfitting risk

    Untuk simplicity, kita pakai raw normalized landmarks dulu.

===============================================
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json

# ============================================
# KONFIGURASI
# ============================================

BASE_DIR = Path(__file__).parent.parent
CROPPED_DIR = BASE_DIR / "dataset" / "cropped"
LANDMARKS_DIR = BASE_DIR / "dataset" / "landmarks"
STATS_FILE = LANDMARKS_DIR / "extraction_stats.json"

# MediaPipe configuration
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # 🔥 EKSTRAKSI 2 TANGAN untuk huruf yang memerlukan 2 tangan
    min_detection_confidence=0.5,
)

# Class names
CLASS_NAMES = [chr(65 + i) for i in range(26)]  # A-Z


# ============================================
# LANDMARK EXTRACTION
# ============================================


def extract_landmarks(image_path):
    """
    Extract MediaPipe hand landmarks dari image (hingga 2 tangan).

    📚 KONSEP: Geometric Feature Extraction (2-HAND SUPPORT)

    MediaPipe Hands output:
    - 21 landmarks × (x, y, z) per tangan
    - x, y: Normalized 0-1 (image coordinates)
    - z: Depth relative to wrist (real-world scale)

    Normalization Strategy:
    1. Detect landmarks (raw coordinates)
    2. Normalize relative to wrist (landmark 0)
    3. Scale by bounding box size
    4. Result: Translation & scale invariant!

    🔥 2-HAND SUPPORT:
    - Jika 2 tangan: 126 features (63 × 2)
    - Jika 1 tangan: 126 features (63 + 63 zeros padding)
    - Jika 0 tangan: None (gagal)

    Returns:
        numpy array: (126,) flattened landmarks or None if failed
    """

    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands_detector.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        num_hands = len(results.multi_hand_landmarks)

        # 🔥 EKSTRAKSI HINGGA 2 TANGAN
        all_normalized_landmarks = []

        for hand_idx in range(min(2, num_hands)):  # Maksimal 2 tangan
            hand_landmarks = results.multi_hand_landmarks[hand_idx]

            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Convert to numpy array
            landmarks = np.array(landmarks, dtype=np.float32)

            # Normalize relative to wrist (landmark 0)
            # Wrist coordinates
            wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]

            # Get bounding box for scaling
            x_coords = landmarks[0::3]  # Every 3rd element starting from 0 (x values)
            y_coords = landmarks[1::3]  # Every 3rd element starting from 1 (y values)

            bbox_width = np.max(x_coords) - np.min(x_coords)
            bbox_height = np.max(y_coords) - np.min(y_coords)

            # Avoid division by zero
            if bbox_width < 1e-6 or bbox_height < 1e-6:
                # Jika tangan ini invalid, skip
                all_normalized_landmarks.extend([0.0] * 63)
                continue

            # Normalize coordinates
            normalized_landmarks = []
            for i in range(0, len(landmarks), 3):
                x = (landmarks[i] - wrist_x) / bbox_width
                y = (landmarks[i + 1] - wrist_y) / bbox_height
                z = landmarks[i + 2] - wrist_z
                normalized_landmarks.extend([x, y, z])

            all_normalized_landmarks.extend(normalized_landmarks)

        # 🔥 PADDING: Jika hanya 1 tangan terdeteksi, tambahkan 63 zeros untuk tangan kedua
        if num_hands == 1:
            all_normalized_landmarks.extend([0.0] * 63)

        # Pastikan total 126 features (2 × 63)
        return np.array(all_normalized_landmarks[:126], dtype=np.float32)

    except Exception as e:
        return None


# ============================================
# BATCH PROCESSING
# ============================================


def process_dataset():
    """
    Extract landmarks untuk semua training & validation images.

    📚 KONSEP: Preprocessing Pipeline

    Pipeline:
    1. Iterate all images (train & val)
    2. Extract landmarks dengan MediaPipe
    3. Save landmarks as .npy files (efficient)
    4. Log statistics (success rate, failures)

    Output structure:
    dataset/landmarks/
        train/
            A/
                augmented_image_1.npy
                augmented_image_2.npy
                ...
            B/
                ...
        val/
            A/
                ...

    Why .npy?
    - Fast loading (binary format)
    - Numpy native format
    - Small file size
    """

    print("\n" + "=" * 60)
    print("🤚 EXTRACTING MEDIAPIPE HAND LANDMARKS (2-HAND SUPPORT)")
    print("=" * 60)
    print("\n📚 MULTI-MODAL LEARNING CONCEPT:")
    print("   Modality 1: Image pixels (visual features)")
    print("   Modality 2: Hand landmarks (geometric features)")
    print("   → Combine both for robust recognition!")
    print("\n🔥 2-HAND SUPPORT:")
    print("   ✋✋ Ekstraksi hingga 2 tangan per image")
    print("   📊 126 features total (63 × 2 tangan)")
    print("   🎯 Lebih akurat untuk huruf 2-tangan (C, G, H, J, dll)\n")

    stats = {
        "train": {"success": 0, "failed": 0, "per_class": {}},
        "val": {"success": 0, "failed": 0, "per_class": {}},
    }

    failed_files = []

    # Process train & val
    for split in ["train", "val"]:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split...")
        print(f"{'='*60}\n")

        split_dir = CROPPED_DIR / split
        output_dir = LANDMARKS_DIR / split
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each class
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            # Get all images
            image_files = list(class_dir.glob("*.jpg"))

            class_success = 0
            class_failed = 0

            print(f"📂 Class {class_name}: {len(image_files)} images")

            for image_path in tqdm(image_files, desc=f"  {class_name}", unit="img"):
                # Extract landmarks
                landmarks = extract_landmarks(image_path)

                if landmarks is not None:
                    # Save as .npy
                    output_path = output_class_dir / (image_path.stem + ".npy")
                    np.save(output_path, landmarks)

                    class_success += 1
                    stats[split]["success"] += 1
                else:
                    class_failed += 1
                    stats[split]["failed"] += 1
                    failed_files.append(str(image_path))

            stats[split]["per_class"][class_name] = {
                "success": class_success,
                "failed": class_failed,
                "total": len(image_files),
                "success_rate": (
                    class_success / len(image_files) * 100
                    if len(image_files) > 0
                    else 0
                ),
            }

            print(
                f"   ✅ Success: {class_success} | ❌ Failed: {class_failed} | Rate: {class_success/(class_success+class_failed)*100:.1f}%"
            )

    # Save statistics
    stats["failed_files"] = failed_files
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)

    total_success = stats["train"]["success"] + stats["val"]["success"]
    total_failed = stats["train"]["failed"] + stats["val"]["failed"]
    total_images = total_success + total_failed

    print(
        f"\n✅ Total Success: {total_success}/{total_images} ({total_success/total_images*100:.2f}%)"
    )
    print(
        f"❌ Total Failed: {total_failed}/{total_images} ({total_failed/total_images*100:.2f}%)"
    )

    print(f"\n📁 Landmarks saved to: {LANDMARKS_DIR}")
    print(f"📈 Statistics saved to: {STATS_FILE}")

    if total_failed > 0:
        print(f"\n⚠️  {total_failed} images failed landmark extraction")
        print("   Possible reasons:")
        print("   - Hand not clearly visible")
        print("   - Multiple hands in frame")
        print("   - Extreme pose/occlusion")
        print("\n   These images will be skipped during multi-modal training.")
        print("   Single-modal (image-only) training dapat menggunakan semua images.")

    print("\n" + "=" * 60)
    print("🎓 KONSEP YANG DIPELAJARI:")
    print("=" * 60)
    print("✓ Multi-modal learning: Multiple data types")
    print("✓ MediaPipe Hands: 21-point hand landmarks")
    print("✓ Coordinate normalization: Translation & scale invariance")
    print("✓ Feature extraction: Geometric information")
    print("✓ Preprocessing pipeline: Batch data preparation")
    print("✓ Error handling: Graceful failure + logging")
    print("=" * 60)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n🚀 Starting MediaPipe Landmarks Extraction...")
    print("\n📚 WHY MULTI-MODAL?")
    print("   Image alone: Great for visual patterns, but affected by:")
    print("   - Lighting conditions")
    print("   - Camera angle")
    print("   - Distance/scale")
    print("   - Background clutter")
    print("\n   Landmarks add: Geometric structure information")
    print("   - Invariant to lighting")
    print("   - Can be normalized for angle/scale")
    print("   - Explicit hand pose")
    print("   - Complementary to visual features")
    print("\n   Together = More Robust! 🚀\n")

    process_dataset()

    print("\n✅ Landmark extraction complete!")
    print("\nNext steps:")
    print("1. Train single-modal model dengan robust augmentation")
    print("2. Train multi-modal model dengan image + landmarks")
    print("3. Compare performance: Which is better?")
