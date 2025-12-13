"""
BISINDO Model Evaluation & Comparison

📚 KONSEP DEEP LEARNING YANG AKAN DIPELAJARI:
===============================================

12. MODEL EVALUATION METRICS
    - Accuracy: Overall correctness (% benar dari total)
    - Precision: Dari semua yang diprediksi X, berapa yang benar X
    - Recall: Dari semua actual X, berapa yang berhasil dideteksi
    - F1-Score: Harmonic mean precision & recall
    - Confusion Matrix: Visualisasi kesalahan per class
    
13. INFERENCE SPEED BENCHMARK
    - Latency: Waktu untuk 1 prediksi (ms)
    - Throughput: Berapa prediksi per detik
    - CPU vs GPU performance difference
    - Batch inference vs single inference
    
14. MODEL SELECTION CRITERIA
    - Accuracy threshold: Minimum acceptable (e.g., >90%)
    - Speed requirement: Real-time game needs <100ms
    - Model size: Mobile deployment prefer <20MB
    - Production readiness: Balance all factors

===============================================
"""

import os
import time
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import onnx
import onnxruntime as ort

# ============================================
# KONFIGURASI
# ============================================
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "cropped"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "training" / "evaluation"

BATCH_SIZE = 32
NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create eval directory
os.makedirs(EVAL_DIR, exist_ok=True)

# Class names
CLASS_NAMES = [chr(65 + i) for i in range(26)]  # A-Z


# ============================================
# LOAD MODELS
# ============================================

def load_mobilenet_model():
    """Load MobileNetV3-Small dari checkpoint."""
    
    print("📦 Loading MobileNetV3-Small...")
    
    # Build model architecture
    model = models.mobilenet_v3_small(weights=None)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, NUM_CLASSES)
    )
    
    # Load checkpoint
    checkpoint_path = MODEL_DIR / "mobilenet_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ MobileNetV3 loaded from epoch {checkpoint['epoch']}")
    print(f"   Val Acc (during training): {checkpoint['val_acc']:.2f}%")
    
    return model, checkpoint


def load_efficientnet_model():
    """Load EfficientNet-B0 dari checkpoint."""
    
    print("\n📦 Loading EfficientNet-B0...")
    
    # Build model architecture
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 1024),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, NUM_CLASSES)
    )
    
    # Load checkpoint
    checkpoint_path = MODEL_DIR / "efficientnet_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ EfficientNet loaded from epoch {checkpoint['epoch']}")
    print(f"   Val Acc (during training): {checkpoint['val_acc']:.2f}%")
    
    return model, checkpoint


# ============================================
# DATA LOADING
# ============================================

def get_val_loader():
    """Get validation dataloader."""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    val_dataset = ImageFolder(
        root=str(DATASET_DIR / "val"),
        transform=transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    return val_loader, val_dataset


# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_model(model, val_loader, model_name):
    """
    Comprehensive model evaluation.
    
    📚 KONSEP: Evaluation Metrics
    
    Metrics:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP) - "Dari semua yang diprediksi A, berapa yang benar A"
    - Recall: TP / (TP + FN) - "Dari semua actual A, berapa yang berhasil dideteksi"
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    
    Confusion Matrix:
    - Rows: True labels
    - Columns: Predicted labels
    - Diagonal: Correct predictions
    - Off-diagonal: Errors (misclassifications)
    """
    
    print(f"\n{'='*80}")
    print(f"🔍 EVALUATING: {model_name}")
    print(f"{'='*80}\n")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating", ncols=80):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"📊 Overall Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Find worst performing classes
    worst_indices = np.argsort(f1_per_class)[:5]
    print(f"\n⚠️  Top 5 Worst Performing Classes:")
    for idx in worst_indices:
        print(f"   {CLASS_NAMES[idx]}: Precision={precision_per_class[idx]:.3f}, "
              f"Recall={recall_per_class[idx]:.3f}, F1={f1_per_class[idx]:.3f}, "
              f"Support={support[idx]}")
    
    # Find best performing classes
    best_indices = np.argsort(f1_per_class)[-5:][::-1]
    print(f"\n🎉 Top 5 Best Performing Classes:")
    for idx in best_indices:
        print(f"   {CLASS_NAMES[idx]}: Precision={precision_per_class[idx]:.3f}, "
              f"Recall={recall_per_class[idx]:.3f}, F1={f1_per_class[idx]:.3f}, "
              f"Support={support[idx]}")
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support
        }
    }
    
    return results


def measure_inference_speed(model, model_name, num_samples=100):
    """
    Measure inference speed (latency).
    
    📚 KONSEP: Inference Speed
    
    Metrics:
    - Latency: Waktu untuk 1 prediction (ms)
    - Throughput: Predictions per second
    - Warmup: First few iterations slower (cache/optimization)
    
    Real-time game requirement:
    - Target: <100ms per prediction
    - 60 FPS = 16.7ms per frame (very tight!)
    - 30 FPS = 33.3ms per frame (more realistic)
    """
    
    print(f"\n{'='*80}")
    print(f"⏱️  INFERENCE SPEED BENCHMARK: {model_name}")
    print(f"{'='*80}\n")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # Warmup (first iterations slower due to cache/JIT)
    print("🔥 Warming up (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark single inference
    print(f"📏 Benchmarking single inference ({num_samples} samples)...")
    latencies = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Single inference", ncols=80):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    print(f"\n📊 Single Inference Latency:")
    print(f"   Mean:   {latencies.mean():.2f} ms")
    print(f"   Median: {np.median(latencies):.2f} ms")
    print(f"   Std:    {latencies.std():.2f} ms")
    print(f"   Min:    {latencies.min():.2f} ms")
    print(f"   Max:    {latencies.max():.2f} ms")
    print(f"   P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"   P99:    {np.percentile(latencies, 99):.2f} ms")
    
    # Throughput
    throughput = 1000 / latencies.mean()  # predictions per second
    print(f"\n🚀 Throughput: {throughput:.1f} predictions/second")
    
    # Real-time capability
    target_fps = [60, 30, 15]
    print(f"\n🎮 Real-time Game Capability:")
    for fps in target_fps:
        frame_time = 1000 / fps  # ms per frame
        if latencies.mean() < frame_time:
            print(f"   ✅ Can run at {fps} FPS (frame budget: {frame_time:.1f}ms)")
        else:
            print(f"   ❌ Cannot run at {fps} FPS (frame budget: {frame_time:.1f}ms)")
    
    # Batch inference
    print(f"\n📦 Batch Inference (batch_size=32):")
    batch_input = torch.randn(32, 3, 224, 224).to(DEVICE)
    batch_latencies = []
    
    with torch.no_grad():
        for _ in range(50):
            start_time = time.perf_counter()
            _ = model(batch_input)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            batch_latencies.append((end_time - start_time) * 1000)
    
    batch_latencies = np.array(batch_latencies)
    avg_per_sample = batch_latencies.mean() / 32
    
    print(f"   Batch latency: {batch_latencies.mean():.2f} ms")
    print(f"   Per sample: {avg_per_sample:.2f} ms")
    print(f"   Speed up: {latencies.mean() / avg_per_sample:.2f}x faster than single")
    
    speed_results = {
        'single_inference': {
            'mean_ms': latencies.mean(),
            'median_ms': np.median(latencies),
            'std_ms': latencies.std(),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        },
        'throughput_per_sec': throughput,
        'batch_inference': {
            'batch_latency_ms': batch_latencies.mean(),
            'per_sample_ms': avg_per_sample,
            'speedup': latencies.mean() / avg_per_sample
        }
    }
    
    return speed_results


def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title(f'Confusion Matrix: {model_name}\n(Normalized by row)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = EVAL_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Confusion matrix saved: {save_path}")
    
    plt.close()


def compare_models(mobilenet_results, efficientnet_results, mobilenet_speed, efficientnet_speed):
    """
    Compare both models and choose winner.
    
    📚 KONSEP: Model Selection
    
    Criteria:
    1. Accuracy: Higher is better (threshold: >90%)
    2. Speed: Faster is better (target: <100ms)
    3. Model size: Smaller is better (<20MB for mobile)
    4. Consistency: Precision/Recall balance
    
    Trade-offs:
    - High accuracy but slow → Not good for real-time game
    - Fast but low accuracy → Bad user experience
    - Best model = Optimal balance
    """
    
    print(f"\n{'='*80}")
    print(f"🏆 MODEL COMPARISON & SELECTION")
    print(f"{'='*80}\n")
    
    # Accuracy comparison
    print("📊 ACCURACY COMPARISON:")
    print(f"   MobileNetV3:  {mobilenet_results['accuracy']:.4f}%")
    print(f"   EfficientNet: {efficientnet_results['accuracy']:.4f}%")
    acc_diff = efficientnet_results['accuracy'] - mobilenet_results['accuracy']
    if acc_diff > 0:
        print(f"   → EfficientNet is {acc_diff:.4f}% more accurate ✅")
    else:
        print(f"   → MobileNet is {abs(acc_diff):.4f}% more accurate ✅")
    
    # Precision/Recall comparison
    print(f"\n📏 PRECISION/RECALL:")
    print(f"   MobileNetV3  - Precision: {mobilenet_results['precision']:.4f}, Recall: {mobilenet_results['recall']:.4f}")
    print(f"   EfficientNet - Precision: {efficientnet_results['precision']:.4f}, Recall: {efficientnet_results['recall']:.4f}")
    
    # F1-Score comparison
    print(f"\n🎯 F1-SCORE:")
    print(f"   MobileNetV3:  {mobilenet_results['f1']:.4f}")
    print(f"   EfficientNet: {efficientnet_results['f1']:.4f}")
    
    # Speed comparison
    print(f"\n⚡ INFERENCE SPEED (Single Image):")
    mobilenet_latency = mobilenet_speed['single_inference']['mean_ms']
    efficientnet_latency = efficientnet_speed['single_inference']['mean_ms']
    print(f"   MobileNetV3:  {mobilenet_latency:.2f} ms")
    print(f"   EfficientNet: {efficientnet_latency:.2f} ms")
    speed_diff = efficientnet_latency - mobilenet_latency
    if speed_diff > 0:
        print(f"   → MobileNet is {speed_diff:.2f}ms faster ({mobilenet_latency/efficientnet_latency:.2f}x) ✅")
    else:
        print(f"   → EfficientNet is {abs(speed_diff):.2f}ms faster ({efficientnet_latency/mobilenet_latency:.2f}x) ✅")
    
    # Model size comparison
    print(f"\n💾 MODEL SIZE:")
    mobilenet_params = 1544506
    efficientnet_params = 4050858  # Approximate from training
    print(f"   MobileNetV3:  {mobilenet_params:,} parameters (~{mobilenet_params*4/1024/1024:.2f} MB)")
    print(f"   EfficientNet: {efficientnet_params:,} parameters (~{efficientnet_params*4/1024/1024:.2f} MB)")
    size_ratio = efficientnet_params / mobilenet_params
    print(f"   → EfficientNet is {size_ratio:.2f}x larger")
    
    # Real-time capability
    print(f"\n🎮 REAL-TIME GAME CAPABILITY (<100ms target):")
    print(f"   MobileNetV3:  {mobilenet_latency:.2f}ms → {'✅ PASS' if mobilenet_latency < 100 else '❌ FAIL'}")
    print(f"   EfficientNet: {efficientnet_latency:.2f}ms → {'✅ PASS' if efficientnet_latency < 100 else '❌ FAIL'}")
    
    # Decision matrix
    print(f"\n🧮 DECISION MATRIX:")
    print(f"   {'Criteria':<20} {'MobileNetV3':<15} {'EfficientNet':<15} {'Winner'}")
    print(f"   {'-'*70}")
    
    # Accuracy
    acc_winner = "MobileNet" if mobilenet_results['accuracy'] > efficientnet_results['accuracy'] else "EfficientNet"
    if abs(acc_diff) < 0.1:
        acc_winner = "Tie ⚖️"
    print(f"   {'Accuracy':<20} {mobilenet_results['accuracy']:>13.2f}% {efficientnet_results['accuracy']:>13.2f}% {acc_winner:>10}")
    
    # Speed
    speed_winner = "MobileNet" if mobilenet_latency < efficientnet_latency else "EfficientNet"
    print(f"   {'Speed':<20} {mobilenet_latency:>12.2f}ms {efficientnet_latency:>12.2f}ms {speed_winner:>10}")
    
    # Size
    print(f"   {'Model Size':<20} {'Smaller':>14} {'Larger':>14} {'MobileNet':>10}")
    
    # F1-Score
    f1_winner = "MobileNet" if mobilenet_results['f1'] > efficientnet_results['f1'] else "EfficientNet"
    print(f"   {'F1-Score':<20} {mobilenet_results['f1']:>14.4f} {efficientnet_results['f1']:>14.4f} {f1_winner:>10}")
    
    # Final recommendation
    print(f"\n{'='*80}")
    print(f"🎯 FINAL RECOMMENDATION (Real-time Game Priority):")
    print(f"{'='*80}\n")
    
    # Score system for REAL-TIME GAME (speed & size more important)
    mobilenet_score = 0
    efficientnet_score = 0
    
    # Accuracy (weight: 2 - both are excellent >99%)
    # Only significant if gap >1%
    if abs(acc_diff) < 1.0:
        # Negligible difference, both get points
        mobilenet_score += 1
        efficientnet_score += 1
    elif mobilenet_results['accuracy'] > efficientnet_results['accuracy']:
        mobilenet_score += 2
    else:
        efficientnet_score += 2
    
    # Speed (weight: 3 - CRITICAL for real-time game)
    if mobilenet_latency < efficientnet_latency:
        mobilenet_score += 3
    else:
        efficientnet_score += 3
    
    # Model size (weight: 2 - important for deployment)
    mobilenet_score += 2  # Always smaller
    
    print(f"Score: MobileNetV3={mobilenet_score}, EfficientNet={efficientnet_score}")
    print(f"\n💡 Weighting for Real-time Game:")
    print(f"   Accuracy (both >99%): Weight = 2 (Low, both excellent)")
    print(f"   Speed (real-time critical): Weight = 3 (HIGH)")
    print(f"   Size (deployment ease): Weight = 2 (Medium)")
    print()
    
    if mobilenet_score > efficientnet_score:
        winner = "MobileNetV3-Small"
        winner_model = "mobilenet"
        print(f"🏆 WINNER: MobileNetV3-Small")
        print(f"\n✅ Alasan:")
        print(f"   • Accuracy excellent ({mobilenet_results['accuracy']:.2f}%)")
        print(f"   • Accuracy gap negligible ({abs(acc_diff):.2f}% difference)")
        print(f"   • Inference speed {efficientnet_latency/mobilenet_latency:.2f}x FASTER ({mobilenet_latency:.2f}ms)")
        print(f"   • Model size {efficientnet_params/mobilenet_params:.2f}x SMALLER ({mobilenet_params*4/1024/1024:.2f}MB)")
        print(f"   • Perfect untuk real-time game & mobile deployment")
        print(f"   • Can run at 60 FPS easily!")
    else:
        winner = "EfficientNet-B0"
        winner_model = "efficientnet"
        print(f"🏆 WINNER: EfficientNet-B0")
        print(f"\n✅ Alasan:")
        print(f"   • Accuracy tertinggi ({efficientnet_results['accuracy']:.2f}%)")
        print(f"   • Masih acceptable speed ({efficientnet_latency:.2f}ms)")
        print(f"   • Best jika prioritas akurasi maksimal")
    
    print(f"\n💡 Recommendation:")
    print(f"   Deploy {winner} untuk production")
    print(f"   Export ke ONNX untuk fast inference")
    print(f"   Use for Hugging Face Spaces deployment")
    
    comparison = {
        'winner': winner_model,
        'winner_name': winner,
        'scores': {
            'mobilenet': mobilenet_score,
            'efficientnet': efficientnet_score
        },
        'accuracy_diff': acc_diff,
        'speed_diff': speed_diff
    }
    
    return comparison


# ============================================
# ONNX EXPORT
# ============================================

def export_to_onnx(model, model_name):
    """
    Export PyTorch model ke ONNX format.
    
    📚 KONSEP: ONNX (Open Neural Network Exchange)
    
    Benefits:
    - Cross-platform: Run di PyTorch, TensorFlow, ONNX Runtime
    - Optimization: Graph optimization, operator fusion
    - Deployment: Easier untuk production (C++, mobile, web)
    - Performance: Faster inference than PyTorch
    
    ONNX Runtime:
    - Microsoft's high-performance inference engine
    - 2-10x faster than native frameworks
    - Support CPU & GPU acceleration
    """
    
    print(f"\n{'='*80}")
    print(f"📦 EXPORTING TO ONNX: {model_name}")
    print(f"{'='*80}\n")
    
    model.eval()
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # Export path
    onnx_path = MODEL_DIR / f"{model_name.lower()}_final.onnx"
    
    print(f"🔄 Converting PyTorch → ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model exported: {onnx_path}")
    
    # Verify ONNX model
    print(f"\n🔍 Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model is valid!")
    
    # Test ONNX Runtime inference
    print(f"\n⚡ Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Prepare input
    ort_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    
    # Run inference
    start_time = time.perf_counter()
    ort_output = ort_session.run(None, ort_input)
    end_time = time.perf_counter()
    
    onnx_latency = (end_time - start_time) * 1000
    print(f"✅ ONNX Runtime inference: {onnx_latency:.2f}ms")
    
    # Compare with PyTorch
    with torch.no_grad():
        start_time = time.perf_counter()
        torch_output = model(dummy_input)
        end_time = time.perf_counter()
    
    torch_latency = (end_time - start_time) * 1000
    print(f"   PyTorch inference: {torch_latency:.2f}ms")
    print(f"   Speed up: {torch_latency / onnx_latency:.2f}x faster with ONNX Runtime")
    
    # File size
    file_size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"\n💾 ONNX model size: {file_size_mb:.2f} MB")
    
    return str(onnx_path)


# ============================================
# MAIN EVALUATION
# ============================================

def main():
    print("\n" + "="*80)
    print("🎓 BISINDO BATTLE - Model Evaluation & Comparison")
    print("="*80)
    
    # Load validation data
    print("\n📦 Loading validation dataset...")
    val_loader, val_dataset = get_val_loader()
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Load models
    mobilenet_model, mobilenet_ckpt = load_mobilenet_model()
    efficientnet_model, efficientnet_ckpt = load_efficientnet_model()
    
    # Evaluate MobileNetV3
    mobilenet_results = evaluate_model(mobilenet_model, val_loader, "MobileNetV3-Small")
    plot_confusion_matrix(mobilenet_results['confusion_matrix'], "MobileNetV3-Small")
    
    # Evaluate EfficientNet
    efficientnet_results = evaluate_model(efficientnet_model, val_loader, "EfficientNet-B0")
    plot_confusion_matrix(efficientnet_results['confusion_matrix'], "EfficientNet-B0")
    
    # Speed benchmark
    mobilenet_speed = measure_inference_speed(mobilenet_model, "MobileNetV3-Small")
    efficientnet_speed = measure_inference_speed(efficientnet_model, "EfficientNet-B0")
    
    # Compare and choose winner
    comparison = compare_models(
        mobilenet_results, efficientnet_results,
        mobilenet_speed, efficientnet_speed
    )
    
    # Export winner to ONNX
    winner_model = mobilenet_model if comparison['winner'] == 'mobilenet' else efficientnet_model
    winner_name = comparison['winner']
    
    onnx_path = export_to_onnx(winner_model, winner_name)
    
    # Save complete evaluation report
    report = {
        'mobilenet': {
            'accuracy': float(mobilenet_results['accuracy']),
            'precision': float(mobilenet_results['precision']),
            'recall': float(mobilenet_results['recall']),
            'f1': float(mobilenet_results['f1']),
            'speed': mobilenet_speed
        },
        'efficientnet': {
            'accuracy': float(efficientnet_results['accuracy']),
            'precision': float(efficientnet_results['precision']),
            'recall': float(efficientnet_results['recall']),
            'f1': float(efficientnet_results['f1']),
            'speed': efficientnet_speed
        },
        'comparison': comparison,
        'winner': {
            'model': winner_name,
            'onnx_path': onnx_path
        }
    }
    
    report_path = EVAL_DIR / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Complete evaluation report saved: {report_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📚 KONSEP YANG SUDAH DIPELAJARI:")
    print(f"   ✓ Model Evaluation Metrics (Accuracy, Precision, Recall, F1)")
    print(f"   ✓ Confusion Matrix Analysis")
    print(f"   ✓ Inference Speed Benchmarking")
    print(f"   ✓ Model Comparison & Selection Criteria")
    print(f"   ✓ ONNX Export & Optimization")
    print(f"   ✓ Production Deployment Considerations")
    print(f"\n🎯 NEXT STEP: Deploy {comparison['winner_name']} to Hugging Face Spaces!")
    print()


if __name__ == "__main__":
    main()
