# 📝 BISINDO BATTLE Game - Changelog

## [v1.1.0] - 2024-12-16

### ✨ Added

- **Fullscreen Mode**: Press F11 to toggle between fullscreen and windowed mode
  - Automatically adjusts to your screen resolution
  - Displays current resolution in menu footer

### 🐛 Fixed

- **Landmark Visualization**: Fixed debug mode landmark display
  - Landmarks now correctly positioned on webcam feed
  - Added 21-point hand skeleton visualization with connections
  - Improved landmark size (3px → 5px circles) for better visibility
  - Green bounding box around detected hand
  - White lines connecting landmarks (hand skeleton)
  - Red dots marking each landmark point

### 🔧 Changed

- Updated menu footer to show fullscreen shortcut (F11)
- Improved debug mode visualization with full hand skeleton
- Enhanced coordinate transformation for landmark overlay

### 📖 Documentation

- Updated README.md with F11 fullscreen control
- Updated QUICKSTART.md with fullscreen shortcut
- Added CHANGELOG.md for version tracking

---

## [v1.0.0] - 2024-12-15

### 🎉 Initial Release

- 3 Game modes: Challenge, Practice, Time Attack
- Real-time BISINDO sign language recognition (99.94% accuracy)
- Multi-modal AI model (Image + MediaPipe landmarks)
- Interactive gameplay with instant feedback
- Scoring system based on confidence
- Full keyboard navigation
- Debug mode with landmark visualization
- Pause/resume functionality
- Results screen with statistics
- Material Design UI (dark theme)

---

## 🎯 How to Use Debug Mode

1. Press **D** to toggle debug mode ON/OFF
2. When ON, you'll see:
   - 🟢 **Green box**: Hand bounding box
   - 🔴 **Red dots**: 21 landmark points
   - ⚪ **White lines**: Hand skeleton connections
3. This helps you understand:
   - Is hand detected correctly?
   - Are landmarks accurate?
   - Why prediction might be wrong

## 📐 Fullscreen Tips

- **F11**: Toggle fullscreen anytime (even during gameplay)
- **Resolution**: Automatically adapts to your monitor
- **Performance**: May improve frame rate on some systems
- **Exit**: Press F11 again to return to windowed mode

---

## 🐛 Known Issues

### Non-Critical Warnings (Expected)

- `pkg_resources deprecated`: Cosmetic warning from Pygame
- `Feedback manager`: MediaPipe warning for ONNX models (doesn't affect functionality)
- `Landmark projection`: MediaPipe warning for square ROI (works correctly)

### Limitations

- Single hand detection only
- Requires good lighting for optimal accuracy
- Landmarks extraction: ~74% success rate (varies by gesture)

---

## 🔮 Planned Features

### v1.2.0 (Next)

- [ ] Sound effects (correct/wrong/countdown)
- [ ] Background music with volume control
- [ ] Achievement badges system
- [ ] Tutorial mode (guided learning)

### v1.3.0

- [ ] Custom gesture training
- [ ] Save/load progress
- [ ] Statistics dashboard
- [ ] Multiple themes (light/dark/colorblind)

### v2.0.0 (Future)

- [ ] Word recognition (multi-letter)
- [ ] Sentence/phrase support
- [ ] Multiplayer mode (split screen)
- [ ] Online leaderboard

---

## 📊 Version History Summary

| Version | Date       | Changes Summary                   |
| ------- | ---------- | --------------------------------- |
| v1.1.0  | 2024-12-16 | Fullscreen mode + Fixed landmarks |
| v1.0.0  | 2024-12-15 | Initial release with 3 game modes |

---

**Feedback?** Found a bug or have a suggestion? Open an issue on GitHub!

**Repository**: https://github.com/falihdzakwanz/bisindo-battle
