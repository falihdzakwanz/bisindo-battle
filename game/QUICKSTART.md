# 🎮 BISINDO BATTLE - Quick Reference Card

## 🎯 Game Modes

| Mode            | Duration   | Goal                | Difficulty  |
| --------------- | ---------- | ------------------- | ----------- |
| **Challenge**   | 10 rounds  | Complete all rounds | ⭐⭐ Medium |
| **Practice**    | 3 minutes  | Learn freely        | ⭐ Easy     |
| **Time Attack** | 60 seconds | Max score           | ⭐⭐⭐ Hard |

## ⌨️ Controls

### Navigation

- **↑ ↓** : Navigate menu (vertical)
- **← →** : Navigate options (horizontal)
- **ENTER** : Select / Confirm
- **ESC** : Back / Pause / Menu

### Gameplay

- **SPACE** : Submit gesture
- **D** : Toggle debug mode
- **F11** : Toggle fullscreen
- **ESC** : Pause game

## 🎨 Color Meanings

| Color         | Meaning                           |
| ------------- | --------------------------------- |
| 🔵 **Blue**   | Selected, Primary, Info           |
| 🟢 **Green**  | Correct, Success, High confidence |
| 🔴 **Red**    | Wrong, Error, Time running out    |
| 🟡 **Yellow** | Warning, Medium confidence        |
| ⚪ **White**  | Text, Normal state                |
| ⚫ **Gray**   | Disabled, Not detected            |

## 🏆 Scoring

```
Score = Confidence × 100

Example:
95% confidence = 95 points
82% confidence = 82 points
<80% confidence = Rejected (tidak diterima)
```

**Tips**: Higher confidence = higher score!

## 💡 Debug Mode (Press D)

When debug mode is ON, you'll see:

- 🟢 **Green Box**: Hand bounding box
- 🔴 **Red Dots**: 21 hand landmarks
- 📊 **Confidence Bar**: Real-time prediction confidence
- ✅ **Status**: Hand detected / not detected

**Use debug mode to**:

- Learn optimal hand position
- Troubleshoot detection issues
- Understand landmark placement
- Optimize your gestures

## 📏 Optimal Setup

### Distance

- **Ideal**: 30-50cm from webcam
- Too close: Hand too large, cut off
- Too far: Details lost, accuracy drops

### Lighting

- ✅ Good: Front lighting, even
- ⚠️ Acceptable: Slight backlight
- ❌ Bad: Dark room, strong backlight

### Background

- ✅ Best: Plain wall (white, beige)
- ⚠️ OK: Simple background
- ❌ Avoid: Busy patterns, clutter

### Hand Position

- ✅ Center of frame
- ✅ Fingers clearly visible
- ✅ Consistent hand size
- ❌ Avoid partial hand
- ❌ Avoid multiple hands

## 🎓 Learning Strategy

### Beginner (First Time)

1. Start with **Practice Mode**
2. Enable **Debug Mode** (press D)
3. Learn each letter slowly
4. Focus on **accuracy** not speed
5. Watch the landmarks placement

### Intermediate

1. Switch to **Challenge Mode**
2. Complete 10 rounds consistently
3. Aim for >80% accuracy
4. Disable debug mode
5. Build muscle memory

### Advanced

1. Try **Time Attack** mode
2. Go for high scores
3. Optimize gesture transitions
4. Speed + accuracy balance
5. Compete with friends!

## 🚨 Troubleshooting

### "Tidak ada tangan terdeteksi"

- Check lighting (add more light)
- Move hand to center of frame
- Ensure palm facing camera
- Distance: 30-50cm optimal
- Try simpler background

### Low Confidence (<80%)

- Make gesture more clear
- Hold position steadier
- Check finger separation
- Reference BISINDO guide
- Use debug mode to see landmarks

### Game Lags / Stutters

- Close other heavy applications
- Reduce webcam resolution (in code)
- Update graphics drivers
- Check CPU usage
- Restart game

### Wrong Predictions

- Some letters are similar (E vs I)
- Hold gesture for 1-2 seconds
- Make finger positions clear
- Check BISINDO reference
- Practice in Practice Mode

## 📱 Quick Tips

1. **Warm Up**: Practice a few gestures before starting
2. **Consistency**: Use same hand position each time
3. **Patience**: Wait for high confidence before submitting
4. **Debug**: Use D key to understand detection
5. **Breaks**: Take breaks every 10-15 minutes

## 🎯 Achievement Goals

- [ ] Complete first Challenge Mode (10/10)
- [ ] Score 900+ in Challenge Mode (90% avg confidence)
- [ ] Practice Mode: 20+ correct gestures
- [ ] Time Attack: Score 1000+ points
- [ ] Perfect Round: 10/10 with >95% average confidence
- [ ] Speed Master: Time Attack with 15+ gestures in 60s

---

**Have fun learning BISINDO!** 👐🎉

**Need help?** See [README.md](README.md) or [DEVELOPMENT.md](DEVELOPMENT.md)
