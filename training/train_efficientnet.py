"""
BISINDO Training: EfficientNet-B0 with Transfer Learning

📚 KONSEP TAMBAHAN YANG AKAN DIPELAJARI:
===============================================

9. EFFICIENTNET ARCHITECTURE (Compound Scaling)
   - Compound Scaling: Scale width, depth, resolution bersamaan
   - Lebih efisien daripada hanya scale depth (ResNet)
   - Trade-off: Lebih banyak parameters vs akurasi lebih tinggi
   
10. MODEL COMPARISON
    - Fair comparison: Same hyperparameters, same data
    - Metrics: Accuracy, inference speed, model size
    - Choose best model untuk production deployment
    
11. ARCHITECTURE EFFICIENCY
    - Parameters count: Lebih banyak = lebih lambat, butuh lebih banyak memory
    - FLOPs (Floating Point Operations): Computational cost
    - Mobile-friendly vs Server-grade models

===============================================
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================
# KONFIGURASI (IDENTIK DENGAN MOBILENET)
# ============================================
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "cropped"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "training" / "logs" / "efficientnet"
CHECKPOINT_DIR = BASE_DIR / "training" / "checkpoints" / "efficientnet"

# Hyperparameters (SAMA seperti MobileNet untuk fair comparison)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
NUM_CLASSES = 26

# Early stopping & LR scheduling
EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================
# DATA PREPROCESSING (IDENTIK)
# ============================================

def get_data_transforms():
    """Data transforms (sama dengan MobileNet untuk consistency)."""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def load_datasets():
    """Load datasets."""
    
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
    Build EfficientNet-B0 with transfer learning.
    
    📚 KONSEP: EfficientNet vs MobileNet
    
    EfficientNet-B0:
    - Parameters: ~5.3M (lebih banyak dari MobileNet 2.5M)
    - Architecture: MBConv blocks + Squeeze-Excitation
    - Compound scaling: Optimal balance width, depth, resolution
    - Trade-off: Sedikit lebih lambat, tapi lebih akurat
    
    MobileNetV3-Small:
    - Parameters: ~2.5M
    - Architecture: Inverted residuals + Squeeze-Excitation
    - Optimized untuk mobile devices
    - Trade-off: Lebih cepat, model size lebih kecil
    
    Comparison goal:
    - Apakah extra parameters worth it untuk BISINDO?
    - Speed vs accuracy trade-off
    """
    
    print("\n" + "="*60)
    print("🏗️  BUILDING MODEL: EfficientNet-B0")
    print("="*60)
    
    # Load pretrained EfficientNet-B0
    if pretrained:
        print("📦 Loading pretrained weights from ImageNet...")
        model = models.efficientnet_b0(weights='DEFAULT')
        print("✅ Pretrained weights loaded!")
    else:
        print("⚠️  Training from scratch (no pretrained weights)")
        model = models.efficientnet_b0(weights=None)
    
    # Get number of input features for classifier
    num_features = model.classifier[1].in_features
    
    # Replace classifier head
    print(f"\n🔧 Replacing classifier head:")
    print(f"   Original: {num_features} → 1000 classes (ImageNet)")
    print(f"   New:      {num_features} → {num_classes} classes (BISINDO)")
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 1024),
        nn.SiLU(),  # Swish activation (EfficientNet default)
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes)
    )
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size:           ~{total_params*4/1024/1024:.2f} MB (FP32)")
    
    print(f"\n🔍 Comparison with MobileNetV3-Small:")
    print(f"   MobileNet parameters: 1,544,506")
    print(f"   EfficientNet parameters: {total_params:,}")
    print(f"   Difference: {total_params - 1544506:,} more parameters ({(total_params/1544506 - 1)*100:.1f}% larger)")
    
    return model


# ============================================
# TRAINING FUNCTIONS (IDENTIK)
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train one epoch."""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", ncols=100)
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{running_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Validate model."""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"           [Val]  ", ncols=100)
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ============================================
# MAIN TRAINING
# ============================================

def train_model():
    """Main training loop."""
    
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING: BISINDO EfficientNet-B0")
    print("="*80)
    
    # Device info
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
    print(f"   Batch size: {BATCH_SIZE}")
    
    # Build model
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    
    # Loss & optimizer (identik dengan MobileNet)
    print("\n📉 Loss Function: CrossEntropyLoss")
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n⚙️  Optimizer: Adam")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Weight decay: 0.0001")
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )
    
    print(f"\n📊 LR Scheduler: ReduceLROnPlateau")
    print(f"   Patience: {LR_REDUCE_PATIENCE} | Factor: {LR_REDUCE_FACTOR}x")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=LR_REDUCE_PATIENCE,
        factor=LR_REDUCE_FACTOR
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(LOGS_DIR))
    print(f"\n📈 TensorBoard: {LOGS_DIR}")
    
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
    print(f"\n📚 Comparing with MobileNetV3 results:")
    print(f"   MobileNet Best Val Acc: 99.78%")
    print(f"   Can EfficientNet beat this? Let's find out!")
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
            
            # LR scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # TensorBoard
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
                best_model_path = MODEL_DIR / "efficientnet_best.pt"
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
            
            # Compare dengan MobileNet
            mobilenet_best = 99.78
            if val_acc > mobilenet_best:
                diff = val_acc - mobilenet_best
                print(f"  🔥 BEATING MobileNet by {diff:.2f}%!")
            elif val_acc > mobilenet_best - 0.5:
                diff = mobilenet_best - val_acc
                print(f"  ⚡ Close to MobileNet (behind by {diff:.2f}%)")
            
            print(f"{'='*80}\n")
            
            # Checkpoint every 5 epochs
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
                print(f"   Best model: Epoch {best_epoch} - Val Acc {best_val_acc:.2f}%")
                print("="*80)
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
    
    # Training complete
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Training time: {elapsed_time/60:.2f} minutes")
    print(f"🏆 Best model: Epoch {best_epoch}")
    print(f"   Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved: {MODEL_DIR / 'efficientnet_best.pt'}")
    
    # Save history
    history_path = MODEL_DIR / "efficientnet_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Training history saved: {history_path}")
    
    # Plot curves
    plot_training_curves(history)
    
    writer.close()
    
    return model, history


# ============================================
# VISUALIZATION
# ============================================

def plot_training_curves(history):
    """Plot training curves."""
    
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
    
    plot_path = BASE_DIR / "training" / "efficientnet_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training curves saved: {plot_path}")
    
    plt.show()


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎓 BISINDO BATTLE - Deep Learning Training")
    print("   Model: EfficientNet-B0")
    print("   Task: BISINDO Sign Language Classification (A-Z)")
    print("="*80)
    
    train_model()
    
    print("\n📚 KONSEP TAMBAHAN YANG SUDAH DIPELAJARI:")
    print("   ✓ EfficientNet Architecture & Compound Scaling")
    print("   ✓ Model Comparison (Fair benchmarking)")
    print("   ✓ Architecture Efficiency (Parameters vs Performance)")
    print()
    print("🎯 NEXT STEP: Evaluate & Compare kedua model!")
    print("   - Accuracy comparison")
    print("   - Inference speed test")
    print("   - Model size comparison")
    print("   - Choose winner untuk deployment")
    print()
