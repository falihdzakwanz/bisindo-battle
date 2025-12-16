@echo off
REM ============================================
REM 🔥 2-HAND MODEL TRAINING PIPELINE
REM ============================================

echo.
echo ═══════════════════════════════════════════════════════════════
echo   🚀 BISINDO 2-HAND MODEL TRAINING PIPELINE
echo ═══════════════════════════════════════════════════════════════
echo.

REM Aktivasi environment
echo [1/4] 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)
echo ✅ Environment activated!
echo.

REM Step 1: Extract landmarks
echo ═══════════════════════════════════════════════════════════════
echo [2/4] 🤚 Extracting 2-hand landmarks (126 features)...
echo ═══════════════════════════════════════════════════════════════
echo.
python scripts\extract_landmarks.py
if %errorlevel% neq 0 (
    echo ❌ Landmark extraction failed!
    pause
    exit /b 1
)
echo.
echo ✅ Landmarks extracted successfully!
echo.
pause

REM Step 2: Backup old model
echo ═══════════════════════════════════════════════════════════════
echo [3/4] 💾 Backing up old model...
echo ═══════════════════════════════════════════════════════════════
echo.
if exist models\bisindo_multimodal.onnx (
    copy models\bisindo_multimodal.onnx models\bisindo_multimodal_1hand_backup.onnx
    echo ✅ Old model backed up to: models\bisindo_multimodal_1hand_backup.onnx
) else (
    echo ⚠️  No existing model found (first training)
)
echo.
pause

REM Step 3: Train model
echo ═══════════════════════════════════════════════════════════════
echo [4/4] 🧠 Training 2-hand multimodal model...
echo ═══════════════════════════════════════════════════════════════
echo.
echo This will take a while (30+ epochs)...
echo Go grab a coffee ☕
echo.
python training\train_multimodal.py
if %errorlevel% neq 0 (
    echo ❌ Training failed!
    pause
    exit /b 1
)
echo.
echo ✅ Training completed!
echo.

REM Step 4: Export to ONNX
echo ═══════════════════════════════════════════════════════════════
echo 📦 Exporting model to ONNX...
echo ═══════════════════════════════════════════════════════════════
echo.
python scripts\export_multimodal_onnx.py
if %errorlevel% neq 0 (
    echo ❌ ONNX export failed!
    pause
    exit /b 1
)
echo.
echo ✅ Model exported to ONNX!
echo.

REM Done!
echo ═══════════════════════════════════════════════════════════════
echo   ✅ 2-HAND MODEL TRAINING COMPLETE!
echo ═══════════════════════════════════════════════════════════════
echo.
echo 📊 Summary:
echo    • Landmarks: 126 features (2 hands)
echo    • Model: Trained on 2-hand data
echo    • Export: ONNX format ready for game
echo.
echo 🎮 NEXT STEPS:
echo    1. Test model: python game\bisindo_game.py
echo    2. Try 2-hand gestures: C, G, H, J, W, etc.
echo    3. Compare accuracy with old model
echo.
echo 📄 Check training logs in: training\logs\
echo 📈 Check training curves: training\multimodal_history.png
echo.
pause
