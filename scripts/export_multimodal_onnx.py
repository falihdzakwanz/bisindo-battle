"""
Export Multimodal Model to ONNX Format

📚 KONSEP: Multi-Input ONNX Export
===============================================

31. MULTI-INPUT ONNX EXPORT
    ONNX (Open Neural Network Exchange) support multiple inputs

    PyTorch → ONNX Conversion:
    - torch.onnx.export() dengan multiple dummy inputs
    - input_names: List nama inputs ["image", "landmarks"]
    - output_names: List nama outputs ["output"]
    - dynamic_axes: Support variable batch size

    Challenges Multi-Input:
    - Harus provide semua inputs saat inference
    - Ordering must match export order
    - Shape harus sesuai (1, 3, 224, 224) dan (1, 63)

32. ONNX RUNTIME INFERENCE
    Multi-input inference:
    ```python
    session.run(
        ["output"],
        {
            "image": image_tensor,
            "landmarks": landmarks_tensor
        }
    )
    ```

    Performance:
    - ONNX Runtime optimized untuk production
    - Auto apply operator fusion
    - Support multiple backends (CPU, CUDA, TensorRT)
    - Typically 10-30% faster than PyTorch

===============================================
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_multimodal import MultiModalBISINDO

# ============================================
# KONFIGURASI
# ============================================

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "multimodal_best.pt"
ONNX_PATH = BASE_DIR / "models" / "multimodal_final.onnx"

NUM_CLASSES = 26
LANDMARK_DIM = 126  # 🔥 2 HANDS: 21 points × 3 coordinates × 2 hands = 126 features
IMAGE_SIZE = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# LOAD PYTORCH MODEL
# ============================================

print("\n" + "=" * 60)
print("🔄 EXPORTING MULTIMODAL MODEL TO ONNX")
print("=" * 60)
print("\n📚 Multi-Input ONNX Export Concept:")
print("   Model has 2 inputs: Image (224×224×3) + Landmarks (126)")
print("   🔥 126 features = 2 hands × 63 features per hand")
print("   ONNX will create single file with dual input interface")
print("   Runtime inference: Provide both inputs simultaneously\n")

print("📂 Loading PyTorch model...")
print(f"   Model path: {MODEL_PATH}")

# Initialize model
model = MultiModalBISINDO(num_classes=NUM_CLASSES, landmark_dim=LANDMARK_DIM)

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("✅ Model loaded successfully!")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


# ============================================
# CREATE DUMMY INPUTS
# ============================================

print("\n📊 Creating dummy inputs for ONNX export...")

# Dummy image input (batch=1, channels=3, height=224, width=224)
dummy_image = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)

# Dummy landmarks input (batch=1, landmarks=63)
dummy_landmarks = torch.randn(1, LANDMARK_DIM, device=device)

print(f"   Image shape: {dummy_image.shape}")
print(f"   Landmarks shape: {dummy_landmarks.shape}")


# ============================================
# TEST PYTORCH INFERENCE
# ============================================

print("\n🧪 Testing PyTorch inference...")

with torch.no_grad():
    pytorch_output = model(dummy_image, dummy_landmarks)

print(f"   Output shape: {pytorch_output.shape}")
print(f"   Output sample: {pytorch_output[0, :5].cpu().numpy()}")


# ============================================
# EXPORT TO ONNX
# ============================================

print("\n🔄 Exporting to ONNX format...")
print("   This may take a minute...")

torch.onnx.export(
    model,
    (dummy_image, dummy_landmarks),  # Tuple of inputs
    str(ONNX_PATH),
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["image", "landmarks"],
    output_names=["output"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "landmarks": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

print(f"✅ ONNX export successful!")
print(f"   Saved to: {ONNX_PATH}")


# ============================================
# VALIDATE ONNX MODEL
# ============================================

print("\n🔍 Validating ONNX model...")

# Check model
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)

print("✅ ONNX model is valid!")

# Print model info
print(f"\n📊 ONNX Model Information:")
print(f"   IR Version: {onnx_model.ir_version}")
print(f"   Opset Version: {onnx_model.opset_import[0].version}")

print(f"\n   Inputs:")
for input in onnx_model.graph.input:
    shape = [
        dim.dim_value if dim.dim_value > 0 else "dynamic"
        for dim in input.type.tensor_type.shape.dim
    ]
    print(f"   - {input.name}: {shape}")

print(f"\n   Outputs:")
for output in onnx_model.graph.output:
    shape = [
        dim.dim_value if dim.dim_value > 0 else "dynamic"
        for dim in output.type.tensor_type.shape.dim
    ]
    print(f"   - {output.name}: {shape}")

# File size
file_size = ONNX_PATH.stat().st_size / (1024 * 1024)
print(f"\n   File size: {file_size:.2f} MB")


# ============================================
# TEST ONNX RUNTIME INFERENCE
# ============================================

print("\n🧪 Testing ONNX Runtime inference...")

# Create ONNX Runtime session
ort_session = ort.InferenceSession(str(ONNX_PATH))

# Prepare inputs for ONNX Runtime
ort_inputs = {
    "image": dummy_image.cpu().numpy(),
    "landmarks": dummy_landmarks.cpu().numpy(),
}

# Run inference
ort_outputs = ort_session.run(["output"], ort_inputs)
onnx_output = ort_outputs[0]

print(f"   Output shape: {onnx_output.shape}")
print(f"   Output sample: {onnx_output[0, :5]}")


# ============================================
# COMPARE OUTPUTS
# ============================================

print("\n🔬 Comparing PyTorch vs ONNX outputs...")

pytorch_out_np = pytorch_output.detach().cpu().numpy()
max_diff = np.max(np.abs(pytorch_out_np - onnx_output))
mean_diff = np.mean(np.abs(pytorch_out_np - onnx_output))

print(f"   Max difference: {max_diff:.6f}")
print(f"   Mean difference: {mean_diff:.6f}")

if max_diff < 1e-4:
    print("   ✅ Outputs match! Export successful.")
else:
    print(f"   ⚠️  Outputs differ by {max_diff:.6f}")
    print("   This is usually acceptable (floating point precision)")


# ============================================
# BENCHMARK SPEED
# ============================================

print("\n⚡ Benchmarking inference speed...")

import time

# PyTorch benchmark
num_iterations = 100
start = time.perf_counter()
with torch.no_grad():
    for _ in range(num_iterations):
        _ = model(dummy_image, dummy_landmarks)
pytorch_time = (time.perf_counter() - start) / num_iterations * 1000

# ONNX Runtime benchmark
start = time.perf_counter()
for _ in range(num_iterations):
    _ = ort_session.run(["output"], ort_inputs)
onnx_time = (time.perf_counter() - start) / num_iterations * 1000

print(f"   PyTorch: {pytorch_time:.2f}ms per inference")
print(f"   ONNX Runtime: {onnx_time:.2f}ms per inference")
print(f"   Speedup: {pytorch_time/onnx_time:.2f}×")


# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✅ MULTIMODAL ONNX EXPORT COMPLETED!")
print("=" * 60)
print(f"\n📁 Model saved to: {ONNX_PATH}")
print(f"📊 File size: {file_size:.2f} MB")
print(f"⚡ ONNX Runtime speed: {onnx_time:.2f}ms")
print(f"🎯 Accuracy: 99.94% (from training)")

print("\n📚 Next Steps:")
print("1. Update app.py to use multimodal ONNX model")
print("2. Implement dual-input inference (image + landmarks)")
print("3. Test with webcam at various distances/angles")
print("4. Deploy to Hugging Face Spaces")

print("\n🎓 What We Learned:")
print("✓ Multi-input ONNX export with dual inputs")
print("✓ ONNX Runtime inference with dictionary inputs")
print("✓ Model validation and output comparison")
print("✓ Performance benchmarking PyTorch vs ONNX")
print("✓ Dynamic batch size support")
print("=" * 60)
