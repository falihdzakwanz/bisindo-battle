# 🎮 BISINDO BATTLE - Pygame Version

Interactive desktop game untuk belajar Bahasa Isyarat Indonesia (BISINDO) dengan real-time AI recognition!

## ✨ Features

- **3 Game Modes**:

  - 🎯 **Challenge Mode**: 10 rounds berurutan
  - 🎓 **Practice Mode**: Latihan bebas 3 menit
  - ⚡ **Time Attack**: Score maksimal dalam 60 detik

- **Real-time Recognition**: Multi-modal AI (Image + Landmarks) dengan 99.94% accuracy
- **Debug Mode**: Visualisasi MediaPipe landmarks (tekan `D`)
- **Keyboard Controls**: Full keyboard navigation (no mouse required!)
- **Live Feedback**: Instant prediction dengan confidence bar
- **Score System**: Kompetitif scoring berdasarkan confidence

## 🎮 Controls

| Key          | Action                             |
| ------------ | ---------------------------------- |
| **↑↓ or ←→** | Navigate menu/options              |
| **ENTER**    | Select/Confirm                     |
| **SPACE**    | Submit gesture (saat game)         |
| **D**        | Toggle debug mode (show landmarks) |
| **F11**      | **Toggle fullscreen**              |
| **ESC**      | Back/Pause/Menu                    |

## 🚀 Quick Start

### Prerequisites

```bash
# Pastikan sudah install dependencies
pip install -r requirements.txt

# Model ONNX harus ada di models/
# Run training terlebih dahulu atau download dari HF Space
```

### Run Game

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Run game
python game/bisindo_game.py
```

## 📖 How to Play

### Challenge Mode

1. Game akan menampilkan huruf target
2. Tunjukkan gesture BISINDO sesuai huruf
3. Tekan **SPACE** untuk submit ketika confidence tinggi
4. Selesaikan 10 rounds untuk lihat hasil

### Practice Mode

1. Latihan bebas selama 3 menit
2. Huruf target akan berganti otomatis setelah benar
3. Fokus pada akurasi, bukan kecepatan
4. Gunakan debug mode (tekan **D**) untuk lihat landmarks

### Time Attack

1. Score sebanyak mungkin dalam 60 detik!
2. Setiap jawaban benar dapat poin berdasarkan confidence
3. Fast-paced gameplay
4. Kompetitif dan menantang

## 🎯 Tips untuk Score Tinggi

- ✋ **Jarak Optimal**: 30-50cm dari webcam
- 💡 **Lighting**: Pastikan cahaya cukup, tidak backlight
- 🖼️ **Background**: Gunakan background sederhana/polos
- 👌 **Posisi**: Tangan di tengah frame
- 🎯 **Gesture**: Ikuti standar BISINDO dengan jelas
- ⚡ **Confidence**: Submit hanya ketika confidence >80%

## 🐛 Debug Mode

Tekan **D** untuk toggle debug mode:

- 🟢 **Green Box**: Hand detection bounding box
- 🔴 **Red Dots**: 21 MediaPipe hand landmarks
- 📊 **Confidence Bar**: Real-time prediction confidence
- ✅ **Hand Status**: Hand detected/not detected

Debug mode sangat berguna untuk:

- Troubleshooting gesture yang tidak terdeteksi
- Mempelajari posisi landmarks
- Optimasi jarak dan angle
- Understanding model behavior

## 🏆 Scoring System

```python
# Score calculation
score_per_gesture = confidence * 100

# Example:
# Confidence 95% = 95 points
# Confidence 82% = 82 points
# Confidence <80% = rejected (tidak diterima)
```

**Bonus Tips**:

- Higher confidence = higher score
- Konsistensi gesture penting
- Gesture jelas lebih cepat dikenali

## 🎨 UI Design

**Material Design Inspired**:

- Dark theme untuk mengurangi eye strain
- Color coding:
  - 🔵 **Blue**: Primary/Info
  - 🟢 **Green**: Success/Correct
  - 🔴 **Red**: Error/Wrong
  - 🟡 **Yellow**: Warning/Medium confidence

**Responsive Layout**:

- 1280x720 resolution (720p)
- 60 FPS gameplay
- Smooth animations

## 🔧 Troubleshooting

### Webcam tidak terdeteksi

```python
# Check webcam index
cap = cv2.VideoCapture(0)  # Try 0, 1, 2
```

### Model not found error

```bash
# Pastikan model ada di:
models/multimodal_final.onnx
models/multimodal_final.onnx.data

# Download dari training atau HF Space
```

### FPS drop

```python
# Reduce webcam resolution di code:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Default: 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Default: 480
```

### Landmarks tidak terdeteksi

- Pastikan tangan terlihat jelas
- Cek lighting (tidak terlalu gelap)
- Jarak optimal 30-50cm
- Gunakan background kontras dengan tangan

## 📊 Performance

- **Inference Speed**: 1.58ms (ONNX Runtime)
- **Total Latency**: ~30-50ms (termasuk webcam + preprocessing)
- **FPS**: 60 (display) / 30 (webcam)
- **Memory**: ~500MB (model + webcam buffer)

## 🎓 Educational Value

Game ini dirancang untuk:

- **Gamification**: Belajar BISINDO lebih fun
- **Instant Feedback**: Tahu langsung benar/salah
- **Progressive Learning**: Dari practice ke challenge
- **Competitive**: Time attack untuk motivasi
- **Visual Learning**: Debug mode untuk memahami detection

## 🚧 Future Improvements

- [ ] Sound effects & background music
- [ ] Multiplayer mode (split screen)
- [ ] Daily challenges
- [ ] Leaderboard (online)
- [ ] Achievement system
- [ ] Tutorial mode dengan guided learning
- [ ] Word recognition (multi-letter)
- [ ] Custom gesture training

## 📝 Notes

- Game ini menggunakan multi-modal AI (99.94% accuracy)
- Landmark detection dengan MediaPipe Hands
- Real-time inference dengan ONNX Runtime
- Keyboard-only navigation (accessible)
- Offline gameplay (tidak perlu internet)

## 🤝 Contributing

Kontribusi welcome untuk:

- UI/UX improvements
- New game modes
- Sound effects
- Performance optimization
- Bug fixes

## 📄 License

MIT License - See main repo LICENSE

---

**Have fun learning BISINDO! 🎉👐**

Buat masalah atau saran? Open issue di GitHub!
