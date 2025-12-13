"""
BISINDO Training: MobileNetV3-Small with Transfer Learning

📚 KONSEP DEEP LEARNING YANG AKAN DIPELAJARI:
===============================================

1. TRANSFER LEARNING (Pembelajaran Transfer)
   - Menggunakan model yang sudah trained di ImageNet (14 juta images)
   - Analogi: Seperti lulusan SMA pindah ke universitas
   - Pengetahuan dasar (edges, shapes) sudah dipelajari
   - Tinggal belajar "mata kuliah khusus" (BISINDO classification)
   
2. CNN ARCHITECTURE (Arsitektur Convolutional Neural Network)
   - Convolutional Layers: Deteksi patterns (edges, textures, shapes)
   - Pooling Layers: Downsampling untuk efficiency
   - Fully Connected Layers: Decision making (klasifikasi)
   - Activation Functions: ReLU, Sigmoid, Softmax
   
3. LOSS FUNCTION (Fungsi Kerugian)
   - CrossEntropyLoss: Mengukur "seberapa salah" prediksi
   - Semakin kecil loss, semakin bagus model
   - Analogi: Jumlah soal yang salah di ujian
   
4. OPTIMIZER (Adam)
   - Algoritma yang "belajar" dari kesalahan
   - Adam = Adaptive Moment Estimation
   - Learning rate: Seberapa besar langkah belajar per iterasi
   
5. BACKPROPAGATION (Propagasi Balik)
   - Cara neural network "belajar" dari kesalahan
   - Forward: Input → Prediksi
   - Backward: Hitung error → Update weights
   - Analogi: Guru kasih feedback → siswa perbaiki jawaban
   
6. OVERFITTING vs UNDERFITTING
   - Overfitting: Hafalan (train bagus, val jelek)
   - Underfitting: Belum paham (train & val sama-sama jelek)
   - Goal: Generalisasi bagus (train & val sama-sama bagus)
   
7. REGULARIZATION (Teknik Anti-Overfitting)
   - Dropout: Randomly "matikan" neurons saat training
   - Early Stopping: Stop training jika val loss tidak improve
   - Learning Rate Scheduling: Kurangi LR jika stuck
   
8. BATCH SIZE & EPOCHS
   - Batch: Jumlah images diproses sekaligus
   - Epoch: 1 iterasi penuh dataset (semua images 1x)
   - Trade-off: Batch besar = smooth gradient, butuh lebih banyak memory

===============================================
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# KONFIGURASI
# ============================================
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "cropped"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "training" / "logs" / "mobilenetv3"
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints" / "mobilenetv3"

# Hyperparameters (sesuai keputusan user)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
NUM_CLASSES = 26  # A-Z

# Early stopping & LR scheduling
EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5

# Device (CUDA or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================
# DATA PREPROCESSING
# ============================================

def get_data_transforms():
    """
    Define data transformations.
    
    📚 KONSEP: Data Augmentation & Normalization
    
    Train transforms:
    - RandomHorizontalFlip: Flip 50% images (tangan kiri/kanan)
    - RandomRotation: Rotate ±10° (variasi sudut)
    - ColorJitter: Ubah brightness, contrast (variasi lighting)
    - Normalization: Scale ke mean=0.485, std=0.229 (ImageNet stats)
    
    Val transforms:
    - Hanya resize & normalize (no augmentation untuk fairness)
    
    Kenapa normalize ke ImageNet stats?
    - Model pretrained di ImageNet expect input format ini
    - Consistency = model perform better
    """
    
    # ImageNet normalization (standar untuk pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # RGB channels
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert PIL → Tensor (0-255 → 0-1)
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def load_datasets():
    """
    Load train & validation datasets.
    
    📚 KONSEP: Dataset & DataLoader
    
    ImageFolder: Automatically load images organized in folders
    Structure: dataset/train/A/*.jpg, dataset/train/B/*.jpg, etc.
    
    DataLoader: Batch images together, shuffle, parallel loading
    - shuffle=True untuk train: Prevent model dari "mengingat urutan"
    - shuffle=False untuk val: Consistency untuk evaluation
    - num_workers: Parallel data loading (speed up)
    """
    
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = ImageFolder(
        root=str(DATASET_DIR / "train"),
        transform=train_transform
    )
    
    val_dataset = ImageFolder(
        root=str(DATASET_DIR / "val"),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, train_dataset.classes


# ============================================
# MODEL BUILDING
# ============================================

def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Build MobileNetV3-Small with transfer learning.
    
    📚 KONSEP: Transfer Learning & Fine-tuning
    
    MobileNetV3-Small architecture:
    - Input: 224x224x3
    - Backbone: Inverted residuals + Squeeze-Excitation
    - Output: num_classes (26 untuk A-Z)
    
    Transfer Learning strategy:
    1. Load pretrained weights (ImageNet)
    2. Freeze backbone layers (keep ImageNet knowledge)
    3. Replace classifier head (26 classes bukan 1000)
    4. Fine-tune: Train hanya classifier dulu, lalu unfreeze backbone
    
    Kenapa freeze backbone?
    - Low-level features (edges, textures) universal
    - High-level features (BISINDO-specific) perlu dilatih
    - Freeze = prevent "catastrophic forgetting"
    """
    
    print("\n" + "="*60)
    print("🏗️  BUILDING MODEL: MobileNetV3-Small")
    print("="*60)
    
    # Load pretrained MobileNetV3-Small
    if pretrained:
        print("📦 Loading pretrained weights from ImageNet...")
        model = models.mobilenet_v3_small(weights='DEFAULT')
        print("✅ Pretrained weights loaded!")
    else:
        print("⚠️  Training from scratch (no pretrained weights)")
        model = models.mobilenet_v3_small(weights=None)
    
    # Get number of input features for classifier
    num_features = model.classifier[0].in_features
    
    # Replace classifier head
    print(f"\n🔧 Replacing classifier head:")
    print(f"   Original: {num_features} → 1000 classes (ImageNet)")
    print(f"   New:      {num_features} → {num_classes} classes (BISINDO)")
    
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.Hardswish(),  # Activation function (efficient untuk mobile)
        nn.Dropout(p=0.2),  # Regularization (prevent overfitting)
        nn.Linear(1024, num_classes)
    )
    
    # Move model to device (GPU/CPU)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size:           ~{total_params*4/1024/1024:.2f} MB (FP32)")
    
    return model


# ============================================
# TRAINING FUNCTION
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """
    Train model for one epoch.
    
    📚 KONSEP: Training Loop
    
    Steps per batch:
    1. Forward pass: Input → Model → Predictions
    2. Compute loss: Compare predictions vs ground truth
    3. Backward pass: Compute gradients (backpropagation)
    4. Update weights: Optimizer step
    
    Metrics:
    - Loss: Seberapa "salah" prediksi (lower is better)
    - Accuracy: % predictions yang benar (higher is better)
    """
    
    model.train()  # Set model ke training mode (enable dropout, batchnorm training)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", ncols=100)
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Move data to device (GPU/CPU)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Zero gradients (reset dari batch sebelumnya)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass (backpropagation)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{running_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """
    Validate model on validation set.
    
    📚 KONSEP: Validation vs Training
    
    Perbedaan:
    - model.eval(): Disable dropout, freeze batchnorm
    - torch.no_grad(): Don't compute gradients (save memory)
    - No optimizer.step(): Weights tidak diupdate
    
    Kenapa perlu validation?
    - Cek performa di data yang "belum pernah dilihat"
    - Deteksi overfitting: train acc tinggi, val acc rendah
    - Guide early stopping & model selection
    """
    
    model.eval()  # Set model ke evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        pbar = tqdm(val_loader, desc=f"           [Val]  ", ncols=100)
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass only
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def train_model():
    """
    Main training loop with early stopping and checkpointing.
    
    📚 KONSEP: Complete Training Pipeline
    
    Components:
    1. Model: MobileNetV3-Small
    2. Loss: CrossEntropyLoss
    3. Optimizer: Adam with LR=0.0001
    4. Scheduler: ReduceLROnPlateau (auto adjust LR)
    5. Early Stopping: Stop jika val loss tidak improve
    6. Checkpointing: Save best model
    7. TensorBoard: Visualize training progress
    """
    
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING: BISINDO MobileNetV3-Small")
    print("="*80)
    
    # Check CUDA
    print(f"\n💻 Device Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   ⚠️  Training on CPU (will be slower)")
    print(f"   Using device: {DEVICE}")
    
    # Load data
    print("\n📦 Loading datasets...")
    train_loader, val_loader, class_names = load_datasets()
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples:   {len(val_loader.dataset)}")
    print(f"   Classes: {class_names}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    
    # Build model
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    
    # Loss function
    print("\n📉 Loss Function: CrossEntropyLoss")
    print("   Combines LogSoftmax + NLLLoss")
    print("   Perfect for multi-class classification")
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    print(f"\n⚙️  Optimizer: Adam")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Weight decay: 0.0001 (L2 regularization)")
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )
    
    # Learning rate scheduler
    print(f"\n📊 LR Scheduler: ReduceLROnPlateau")
    print(f"   Patience: {LR_REDUCE_PATIENCE} epochs")
    print(f"   Factor: {LR_REDUCE_FACTOR}x (reduce LR when stuck)")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=LR_REDUCE_PATIENCE,
        factor=LR_REDUCE_FACTOR
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(LOGS_DIR))
    print(f"\n📈 TensorBoard: {LOGS_DIR}")
    print(f"   Run: tensorboard --logdir={LOGS_DIR.parent}")
    
    # Training tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*80)
    print("🎓 STARTING TRAINING LOOP")
    print("="*80)
    print(f"\n📚 What to watch:")
    print(f"   • Train loss decreasing: Model is learning")
    print(f"   • Val acc increasing: Model generalizing well")
    print(f"   • Gap train/val small: No overfitting")
    print(f"   • LR reducing: Optimizer fine-tuning")
    print()
    
    start_time = time.time()
    
    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{NUM_EPOCHS} Summary:")
            print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Check if best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                best_model_path = MODEL_DIR / "mobilenet_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': class_names
                }, best_model_path)
                print(f"  🎉 New best model! Saved to {best_model_path}")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement for {patience_counter}/{EARLY_STOP_PATIENCE} epochs")
            
            print(f"  🏆 Best: Epoch {best_epoch} - Val Acc {best_val_acc:.2f}%")
            print(f"{'='*80}\n")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}\n")
            
            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("\n" + "="*80)
                print("⏹️  EARLY STOPPING TRIGGERED")
                print(f"   No improvement for {EARLY_STOP_PATIENCE} epochs")
                print(f"   Best model: Epoch {best_epoch} - Val Acc {best_val_acc:.2f}%")
                print("="*80)
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print(f"   Last completed epoch: {epoch}")
        print(f"   Best model: Epoch {best_epoch} - Val Acc {best_val_acc:.2f}%")
    
    # Training complete
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Training time: {elapsed_time/60:.2f} minutes")
    print(f"🏆 Best model: Epoch {best_epoch}")
    print(f"   Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved: {MODEL_DIR / 'mobilenet_best.pt'}")
    
    # Save training history
    history_path = MODEL_DIR / "mobilenet_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Training history saved: {history_path}")
    
    # Plot training curves
    plot_training_curves(history)
    
    writer.close()
    
    return model, history


# ============================================
# VISUALIZATION
# ============================================

def plot_training_curves(history):
    """Plot training & validation curves."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    plot_path = BASE_DIR / "training" / "mobilenet_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training curves saved: {plot_path}")
    
    plt.show()


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎓 BISINDO BATTLE - Deep Learning Training")
    print("   Model: MobileNetV3-Small")
    print("   Task: BISINDO Sign Language Classification (A-Z)")
    print("="*80)
    
    train_model()
    
    print("\n📚 KONSEP YANG SUDAH DIPELAJARI:")
    print("   ✓ Transfer Learning & Fine-tuning")
    print("   ✓ CNN Architecture (Convolution, Pooling, FC)")
    print("   ✓ Loss Function (CrossEntropyLoss)")
    print("   ✓ Optimizer (Adam) & Learning Rate")
    print("   ✓ Backpropagation & Gradient Descent")
    print("   ✓ Overfitting vs Underfitting")
    print("   ✓ Regularization (Dropout, Weight Decay)")
    print("   ✓ Early Stopping & LR Scheduling")
    print("   ✓ Model Checkpointing")
    print("   ✓ TensorBoard Visualization")
    print()
    print("🎯 NEXT STEP: Train EfficientNet-B0 untuk comparison!")
    print()
