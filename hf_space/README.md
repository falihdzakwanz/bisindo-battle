---
title: BISINDO Battle - Sign Language Recognition
emoji: 👐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# BISINDO BATTLE 🎮👐

Game edukasi interaktif untuk belajar Bahasa Isyarat Indonesia (BISINDO)!

## Features

- 🤖 **AI-Powered Recognition**: MobileNetV3-Small model dengan 99.78% accuracy
- ⚡ **Real-time Detection**: 7.87ms inference speed
- 🎯 **26 Letters**: A-Z BISINDO alphabet
- 📱 **Mobile-Friendly**: Optimized untuk deployment

## Model Performance

- **Architecture**: MobileNetV3-Small
- **Parameters**: 1.5M
- **Accuracy**: 99.78%
- **Speed**: 7.87ms per prediction
- **Size**: 5.89 MB

## Usage

1. Upload gambar tangan dengan gesture BISINDO
2. Atau gunakan webcam untuk real-time prediction
3. Model akan prediksi huruf A-Z dengan confidence score

## Training Details

- **Dataset**: 11,470 BISINDO hand gesture images
- **Training**: Transfer learning dari ImageNet
- **Framework**: PyTorch + ONNX Runtime
- **Optimization**: FP32 precision
