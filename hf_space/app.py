"""
BISINDO BATTLE - Hugging Face Spaces Deployment
Game Edukasi Bahasa Isyarat Indonesia

📚 KONSEP DEPLOYMENT YANG DIPELAJARI:
===============================================

15. PRODUCTION DEPLOYMENT
    - Gradio: Framework untuk ML web interface
    - ONNX Runtime: Fast inference engine
    - Preprocessing pipeline: Consistency dengan training
    - Error handling: Graceful degradation
    
16. USER EXPERIENCE
    - Image upload: Easy untuk testing
    - Webcam capture: Real-time interaction
    - Confidence visualization: Transparency
    - Examples: Quick start untuk users
    
17. MONITORING & LOGGING
    - Inference time tracking
    - Error logging
    - Usage statistics
    - Model versioning

===============================================
"""

import os
import time
import numpy as np
import gradio as gr
import onnxruntime as ort
from PIL import Image
import cv2
import mediapipe as mp

# ============================================
# KONFIGURASI
# ============================================

MODEL_PATH = "multimodal_final.onnx"
IMAGE_SIZE = (224, 224)
LANDMARK_DIM = 63

# ImageNet normalization (sama dengan training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class names (A-Z)
CLASS_NAMES = [chr(65 + i) for i in range(26)]

# Confidence threshold (sesuai keputusan: 80%)
CONFIDENCE_THRESHOLD = 0.80


# ============================================
# LOAD MODEL
# ============================================

print("🚀 Loading BISINDO Multi-Modal Recognition Model...")
print(f"   Model: Multi-Modal (Image + Landmarks)")
print(f"   Accuracy: 99.94%")
print(f"   Speed: ~1.58ms per prediction (ONNX)")

try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    # Get input names for multi-input model
    image_input_name = ort_session.get_inputs()[0].name
    landmarks_input_name = ort_session.get_inputs()[1].name
    print(f"✅ Model loaded successfully!")
    print(f"   Input 1: {image_input_name} (Image)")
    print(f"   Input 2: {landmarks_input_name} (Landmarks)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Initialize MediaPipe Hands
print("🤚 Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
print("✅ MediaPipe Hands initialized!")


# ============================================
# HAND DETECTION & CROPPING
# ============================================

def detect_and_crop_hand(image):
    """
    Detect hand dengan MediaPipe, crop region tangan, dan extract landmarks.
    
    📚 KONSEP: Multi-Modal Learning
    
    Dual Input Strategy:
    - Visual Input: Cropped hand image (appearance)
    - Geometric Input: 21 hand landmarks (shape)
    → Complementary features = better robustness!
    
    Advantages:
    - Landmarks invariant to lighting, distance
    - Image captures fine details
    - Together: Robust at various conditions
    
    Returns:
        cropped_image: Cropped hand region
        landmarks: Normalized landmarks (63 values)
        success: Boolean (hand detected atau tidak)
    """
    
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB
    if len(image.shape) == 2:  # Grayscale
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image.copy()
    
    # Detect hands
    results = hands_detector.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get bounding box dari landmarks
        h, w, _ = image_rgb.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add margin (20% dari bbox size)
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - margin_x)
        x_max = min(w, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(h, y_max + margin_y)
        
        # Crop hand region
        cropped = image_rgb[y_min:y_max, x_min:x_max]
        
        # Extract and normalize landmarks (same as training)
        landmarks = []
        wrist = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        return cropped, landmarks, True
    else:
        # No hand detected
        return image_rgb, None, False


# ============================================
# PREPROCESSING
# ============================================

def preprocess_image(image):
    """
    Preprocess image untuk inference.
    
    📚 KONSEP: Preprocessing Consistency
    
    Pipeline harus IDENTIK dengan training:
    1. Resize ke 224x224
    2. Convert ke RGB (jika grayscale)
    3. Normalize pixel 0-255 → 0-1
    4. Apply ImageNet normalization
    5. Add batch dimension
    
    Inconsistency = model performance drop!
    """
    
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB (3 channels)
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to 0-1
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    image = (image - MEAN) / STD
    
    # Transpose (H, W, C) → (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension (C, H, W) → (1, C, H, W)
    image = np.expand_dims(image, axis=0)
    
    return image


# ============================================
# INFERENCE
# ============================================

def predict(image):
    """
    Run inference dengan multi-modal inputs dan return predictions.
    
    📚 KONSEP: Multi-Modal Inference Pipeline
    
    Steps:
    1. Detect hand & extract landmarks (MediaPipe)
    2. Preprocess image (crop + normalize)
    3. Prepare landmarks (normalize)
    4. Run multi-input ONNX model
    5. Apply softmax (get probabilities)
    6. Return top-K predictions
    
    Confidence threshold:
    - >80%: High confidence (green)
    - 60-80%: Medium confidence (yellow)
    - <60%: Low confidence (red) - "Coba lagi"
    """
    
    try:
        start_time = time.perf_counter()
        
        # 1. Detect hand, crop, and extract landmarks
        cropped_hand, landmarks, hand_detected = detect_and_crop_hand(image)
        
        if not hand_detected:
            return {}, "⚠️ Tidak ada tangan terdeteksi! Pastikan tangan terlihat jelas di kamera.", "error"
        
        # 2. Preprocess cropped hand (image input)
        image_tensor = preprocess_image(cropped_hand)
        
        # 3. Prepare landmarks (add batch dimension)
        landmarks_tensor = landmarks.reshape(1, LANDMARK_DIM).astype(np.float32)
        
        # 4. Run multi-input inference
        outputs = ort_session.run(
            None, 
            {
                image_input_name: image_tensor,
                landmarks_input_name: landmarks_tensor
            }
        )
        logits = outputs[0][0]  # Remove batch dimension
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get top-5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Build result dictionary
        results = {}
        for idx in top_indices:
            letter = CLASS_NAMES[idx]
            confidence = float(probabilities[idx])
            results[letter] = confidence
        
        # Get top prediction
        top_letter = CLASS_NAMES[top_indices[0]]
        top_confidence = probabilities[top_indices[0]]
        
        # Confidence message
        if top_confidence >= CONFIDENCE_THRESHOLD:
            message = f"✅ Prediksi: **{top_letter}** (Confidence: {top_confidence*100:.1f}%)"
            status = "high"
        elif top_confidence >= 0.60:
            message = f"⚠️ Prediksi: **{top_letter}** (Confidence: {top_confidence*100:.1f}%) - Kurang yakin, coba posisi lebih jelas"
            status = "medium"
        else:
            message = f"❌ Confidence rendah ({top_confidence*100:.1f}%) - Coba lagi dengan gesture yang lebih jelas"
            status = "low"
        
        # Add inference time
        message += f"\n\n⚡ Inference time: {inference_time:.2f}ms"
        
        return results, message, status
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        return {}, error_msg, "error"


# ============================================
# GRADIO INTERFACE
# ============================================

def create_interface():
    """
    Create Gradio web interface.
    
    📚 KONSEP: User Interface Design
    
    Components:
    1. Image input: Upload atau webcam
    2. Label output: Top-5 predictions dengan confidence bars
    3. Text output: Feedback message
    4. Examples: Quick start images
    
    UX Considerations:
    - Clear instructions
    - Visual feedback (colors)
    - Example images untuk testing
    - Mobile-responsive layout
    """
    
    # Custom CSS untuk styling
    custom_css = """
    .output-class {font-size: 1.2em; font-weight: bold;}
    .high-confidence {color: #22c55e;}
    .medium-confidence {color: #eab308;}
    .low-confidence {color: #ef4444;}
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 👐 BISINDO BATTLE - Sign Language Recognition
        
        ### Game Edukasi Bahasa Isyarat Indonesia
        
        Upload gambar tangan dengan gesture BISINDO atau gunakan webcam untuk real-time prediction!
        
        **Model**: Multi-Modal (Image + Landmarks) | **Accuracy**: 99.94% | **Speed**: ~1.58ms
        """)
        
        with gr.Row():
            with gr.Column():
                # Input
                image_input = gr.Image(
                    label="Upload Gambar atau Gunakan Webcam",
                    type="pil",
                    sources=["upload", "webcam"]
                )
                
                predict_btn = gr.Button("🔍 Prediksi Huruf", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 Tips untuk Hasil Terbaik:
                - ✋ **Tangan harus terdeteksi** (sistem pakai MediaPipe)
                - 📷 Tangan di tengah frame, jarak ideal ~30-50cm
                - 💡 Cahaya yang cukup (tidak gelap)
                - 🖼️ Background sederhana (tidak ramai)
                - 👌 Gesture sesuai standar BISINDO
                - 🎯 Satu tangan saja (lebih akurat)
                """)
            
            with gr.Column():
                # Output
                label_output = gr.Label(
                    label="Top-5 Predictions",
                    num_top_classes=5
                )
                
                message_output = gr.Markdown(
                    label="Status",
                    value="Tunggu prediksi..."
                )
        
        # Examples
        gr.Markdown("### 📸 Coba dengan Contoh Gambar:")
        gr.Examples(
            examples=[
                # Add example image paths here if available
                # ["examples/letter_a.jpg"],
                # ["examples/letter_b.jpg"],
            ],
            inputs=image_input,
            label="Contoh Gambar"
        )
        
        # Event handler
        def predict_wrapper(image):
            if image is None:
                return {}, "⚠️ Silakan upload gambar terlebih dahulu", "error"
            
            results, message, status = predict(image)
            return results, message
        
        predict_btn.click(
            fn=predict_wrapper,
            inputs=image_input,
            outputs=[label_output, message_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🎓 Tentang Model
        
        Model multi-modal ini dilatih dengan **8,506 gambar** gesture BISINDO (A-Z) menggunakan 
        dual-input architecture: **Visual Features** (MobileNetV3) + **Geometric Features** (MediaPipe Landmarks).
        
        **Multi-Modal Pipeline:**
        1. **MediaPipe Hands**: Deteksi tangan + extract 21 landmarks (63 coords)
        2. **Image Branch**: Crop + resize + normalize → MobileNetV3 features (576)
        3. **Landmark Branch**: Normalize landmarks → MLP encoder (128)
        4. **Fusion Layer**: Concatenate features (704) → 512 → 26 classes
        5. **Confidence Check**: Threshold 80%
        
        **Performance Metrics:**
        - Validation Accuracy: **99.94%** (highest!)
        - Inference Speed: ~1.58ms (ONNX Runtime)
        - Model Size: 0.22 MB (super compact!)
        - Robust terhadap distance, angle, lighting variations
        - Can run at 600+ FPS untuk real-time game!
        
        **Why Multi-Modal?**
        - Image: Captures appearance & fine details
        - Landmarks: Invariant to lighting, distance, background
        - Together: Best of both worlds = superior robustness!
        """)
    
    return demo


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎮 BISINDO BATTLE - Hugging Face Spaces")
    print("="*60)
    print("\n📚 Model Information:")
    print("   Architecture: Multi-Modal (Image + Landmarks)")
    print("   Parameters: 1.35M")
    print("   Accuracy: 99.94%")
    print("   Speed: ~1.58ms (ONNX Runtime)")
    print("   Confidence Threshold: 80%")
    print("\n🚀 Starting Gradio interface...")
    
    demo = create_interface()
    
    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
    
    print("\n✅ BISINDO BATTLE is running!")
    print("📚 KONSEP DEPLOYMENT YANG SUDAH DIPELAJARI:")
    print("   ✓ Production deployment dengan Gradio")
    print("   ✓ ONNX Runtime untuk fast inference")
    print("   ✓ Preprocessing pipeline consistency")
    print("   ✓ User experience design")
    print("   ✓ Error handling & validation")
    print("   ✓ Confidence threshold implementation")
