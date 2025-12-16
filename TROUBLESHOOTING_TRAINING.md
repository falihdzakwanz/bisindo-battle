# 🔧 TROUBLESHOOTING: Training Error Fixed

## ❌ **Error yang Terjadi:**

```
RuntimeError: stack expects each tensor to be equal size, but got [126] at entry 0 and [63] at entry 12
```

## 🔍 **Root Cause Analysis:**

### Problem:

Dataset landmarks **tidak konsisten**:

- **Beberapa file**: 126 features (hasil ekstraksi 2-hand baru)
- **Beberapa file lain**: 63 features (hasil ekstraksi 1-hand lama)

### Why This Happens:

1. Script `extract_landmarks.py` sudah di-update ke 126 features
2. User menjalankan ekstraksi, menghasilkan **beberapa** file .npy baru (126 features)
3. Tetapi file lama (63 features) **masih ada** di folder `dataset/landmarks/`
4. Training loader mencoba batch data → **mixed sizes** → PyTorch error!

### Technical Detail:

```python
# PyTorch DataLoader tries to stack tensors:
batch_landmarks = [
    tensor([126]),  # File baru ✅
    tensor([126]),  # File baru ✅
    ...
    tensor([63]),   # File LAMA ❌ <- MISMATCH!
    ...
]

# torch.stack() FAILS: Cannot stack different sizes!
```

## ✅ **Solution Applied:**

### Step 1: Delete Old Landmarks

```powershell
Remove-Item -Path "dataset\landmarks" -Recurse -Force
```

Hapus SEMUA file landmarks lama untuk memastikan konsistensi.

### Step 2: Re-Extract ALL Landmarks

```bash
python scripts\extract_landmarks.py
```

Generate ulang SEMUA file dengan 126 features (2-hand support).

### Step 3: Continue Training

Setelah ekstraksi selesai, training akan berjalan normal karena SEMUA file punya size yang sama (126 features).

---

## 📊 **Verification:**

Setelah re-ekstraksi, cek consistency:

```python
import numpy as np
from pathlib import Path

landmarks_dir = Path("dataset/landmarks/train/A")
shapes = []

for file in landmarks_dir.glob("*.npy"):
    data = np.load(file)
    shapes.append(data.shape[0])

print(f"Unique shapes: {set(shapes)}")  # Should be: {126}
```

Expected output:

```
Unique shapes: {126}  ✅
```

If you see `{63, 126}` or `{63}`, re-run extraction!

---

## 🎓 **Lessons Learned:**

### 1. **Incremental Updates Don't Work for Batch Data**

Ketika mengubah feature dimension:

- ❌ **WRONG**: Update script → Run ekstraksi → Harapan file baru overwrite lama
- ✅ **RIGHT**: Delete old data → Run full re-extraction → Ensure consistency

### 2. **PyTorch DataLoader Requirements**

DataLoader assumes **homogeneous batch**:

- All tensors in a batch must have **same shape**
- If shapes differ → `torch.stack()` fails

### 3. **Dataset Migration Best Practice**

When changing data format:

```bash
# 1. Backup old data
mv dataset/landmarks dataset/landmarks_backup_63features

# 2. Run new extraction
python scripts/extract_landmarks.py

# 3. Verify new data
# Test a few samples

# 4. Delete backup if all good
rm -rf dataset/landmarks_backup_63features
```

---

## 🚀 **Next Steps After This Fix:**

1. ✅ Wait for re-extraction to complete (~5-10 minutes)
2. ✅ Training will continue automatically
3. ✅ Model will train with 126 features (2-hand support)
4. ✅ Improved accuracy for 2-hand gestures!

---

## ⚠️ **If Error Persists:**

### Check 1: All landmarks are 126 features

```python
import numpy as np
from pathlib import Path

for split in ["train", "val"]:
    for cls in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        path = Path(f"dataset/landmarks/{split}/{cls}")
        if path.exists():
            for file in path.glob("*.npy"):
                data = np.load(file)
                if data.shape[0] != 126:
                    print(f"❌ WRONG SIZE: {file} -> {data.shape}")
```

### Check 2: Model architecture updated

Verify in `training/train_multimodal.py`:

```python
LANDMARK_DIM = 126  # Should be 126, NOT 63!
```

### Check 3: Game inference updated

Verify in `game/bisindo_game.py`:

```python
landmarks_tensor = landmarks.reshape(1, 126)  # NOT (1, 63)!
```

---

## 📈 **Expected Training Output After Fix:**

```
============================================================
Epoch 1/30
------------------------------------------------------------
Train: 100%|████████████████| 500/500 [02:30<00:00, 3.33it/s]
  Train Loss: 2.1234 | Train Acc: 35.2%
  Val Loss: 1.8765 | Val Acc: 42.1%
  ✅ New best model saved!

Epoch 2/30
------------------------------------------------------------
...
```

Training should proceed smoothly without tensor size errors! 🎉
