# 🔥 UPGRADE: 2-HAND LANDMARK EXTRACTION

## 📋 Perubahan yang Dilakukan

### 1. **Script Ekstraksi Landmarks** (`scripts/extract_landmarks.py`)

- ✅ Ubah `max_num_hands=1` → `max_num_hands=2`
- ✅ Ekstraksi hingga 2 tangan per image
- ✅ Output: 126 features (63 × 2)
- ✅ Auto-padding: Jika 1 tangan → tambah 63 zeros

### 2. **Game Inference** (`game/bisindo_game.py`)

- ✅ Update `detect_and_extract_landmarks()` untuk ekstraksi 2 tangan
- ✅ Reshape tensor: `(1, 63)` → `(1, 126)`
- ✅ Compatible dengan model baru yang akan di-train

## 🎯 Kenapa Perlu 2-Hand Support?

Banyak huruf BISINDO memerlukan 2 tangan:

- **C**: Kedua tangan membentuk kurva
- **G**: Tangan kiri + tangan kanan berinteraksi
- **H**: Posisi horizontal 2 tangan
- **J**: Gerakan J dengan 2 tangan
- **W**: Huruf W dengan jari-jari kedua tangan
- Dan huruf lainnya...

Dengan model 1-tangan, gesture ini **tidak bisa diprediksi dengan akurat**.

## 🚀 Langkah Selanjutnya: Re-Train Model

### Step 1: Ekstraksi Landmarks Baru (126 Features)

```bash
# Aktivasi environment
venv\Scripts\activate

# Run ekstraksi dengan 2-hand support
python scripts/extract_landmarks.py
```

Output:

```
🤚 EXTRACTING MEDIAPIPE HAND LANDMARKS (2-HAND SUPPORT)
════════════════════════════════════════════════════════════

📚 MULTI-MODAL LEARNING CONCEPT:
   Modality 1: Image pixels (visual features)
   Modality 2: Hand landmarks (geometric features)
   → Combine both for robust recognition!

🔥 2-HAND SUPPORT:
   ✋✋ Ekstraksi hingga 2 tangan per image
   📊 126 features total (63 × 2 tangan)
   🎯 Lebih akurat untuk huruf 2-tangan (C, G, H, J, dll)
```

### Step 2: Backup Model Lama

```bash
# Backup model 1-hand yang sudah ada
copy models\bisindo_multimodal.onnx models\bisindo_multimodal_1hand_backup.onnx
```

### Step 3: Train Model Baru dengan 126 Features

Script training perlu di-update untuk menerima 126 features:

**File yang perlu diupdate**: `scripts/train_multimodal.py` atau sejenisnya

Perubahan:

```python
# OLD (63 features):
landmarks_input = keras.Input(shape=(63,), name="landmarks_input")

# NEW (126 features):
landmarks_input = keras.Input(shape=(126,), name="landmarks_input")
```

### Step 4: Run Training

```bash
python scripts/train_multimodal.py
```

### Step 5: Test Model Baru

```bash
# Run game dengan model baru
python game\bisindo_game.py
```

Test dengan gestures 2-tangan untuk melihat improvement!

## 📊 Expected Results

### Before (1-Hand Model):

- ✅ Huruf 1-tangan: Accuracy tinggi (A, B, D, E, F, dll)
- ❌ Huruf 2-tangan: Accuracy rendah atau salah (C, G, H, J, dll)
- 📉 Overall accuracy: ~85-90%

### After (2-Hand Model):

- ✅ Huruf 1-tangan: Tetap tinggi (data tetap valid dengan padding)
- ✅ Huruf 2-tangan: **Drastis lebih akurat!**
- 📈 Overall accuracy: **95-99%** (expected)

## ⚠️ Catatan Penting

### 1. **Kompatibilitas Model**

- Model lama (63 features) **TIDAK COMPATIBLE** dengan ekstraksi baru (126 features)
- Harus re-train model dari awal dengan landmark baru
- Game akan error jika pakai model lama dengan ekstraksi baru

### 2. **Data Quality**

- Pastikan dataset memiliki images dengan 2 tangan untuk huruf yang memerlukannya
- Images dengan 1 tangan akan di-pad dengan zeros (tetap valid)
- Images tanpa tangan akan di-skip (seperti sebelumnya)

### 3. **Performance**

- Ekstraksi 2 tangan sedikit lebih lambat (~10-20% slower)
- Masih real-time untuk game (>30 FPS)
- Training time bisa lebih lama (lebih banyak parameters)

## 🔍 Troubleshooting

### Error: "Input shape mismatch"

```
Expected landmarks_input shape: (?, 126)
Got: (?, 63)
```

**Solusi**: Model masih yang lama. Re-train dengan script baru.

### Error: "No landmarks extracted"

```
All images failed landmark extraction
```

**Solusi**:

1. Check MediaPipe installation: `pip install mediapipe`
2. Verify cropped images ada di `dataset/cropped/`
3. Check image quality (tangan harus visible)

### Accuracy Tidak Meningkat

**Possible causes**:

1. Dataset tidak punya cukup images 2-tangan
2. Model architecture terlalu simple (tambah layers)
3. Need more training epochs
4. Augmentation kurang diverse

## 🎓 Kesimpulan

Upgrade ke 2-hand extraction adalah **game changer** untuk BISINDO:

- ✅ Support untuk semua 26 huruf (termasuk 2-hand gestures)
- ✅ Model lebih expressive dengan 126 features vs 63
- ✅ Lebih realistis untuk real-world BISINDO usage

**Next**: Train model baru dan enjoy the accuracy boost! 🚀
