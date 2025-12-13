"""
BISINDO Training: MobileNetV3 dengan Robust Augmentation

📚 KONSEP YANG DIPELAJARI:
===============================================

22. ROBUST DATA AUGMENTATION
    Problem sebelumnya: Model overfit ke training conditions
    - Hanya recognize tangan close-up
    - Gagal dengan sudut miring
    - Sensitif terhadap jarak/scale

    Solution: Augmentation yang simulate real-world variations

    Augmentation Types:

    A. GEOMETRIC AUGMENTATION
       - RandomResizedCrop: Simulate jarak berbeda (zoom in/out)
       - RandomPerspective: Simulate sudut miring (3D rotation)
       - RandomAffine: Combine rotation + scale + translation
       - RandomHorizontalFlip: Left/right hand variation

    B. COLOR AUGMENTATION
       - ColorJitter: Brightness, contrast, saturation, hue
       - RandomGrayscale: Force model tidak bergantung pada color
       - GaussianBlur: Simulate camera out of focus

    C. QUALITY AUGMENTATION
       - GaussianNoise: Simulate low-light conditions
       - RandomErasing: Simulate occlusion (jari tertutup)

    Trade-off:
    - More augmentation = more robust, but slower training
    - Too extreme augmentation = model confused, accuracy drop
    - Balance is key!

23. SCALE INVARIANCE
    RandomResizedCrop(224, scale=(0.5, 1.0))

    Meaning:
    - scale=0.5: Crop 50% dari image → tangan terlihat besar (close-up)
    - scale=1.0: Crop 100% dari image → tangan terlihat kecil (far away)

    Result: Model belajar recognize tangan di berbagai ukuran!

24. PERSPECTIVE TRANSFORM
    RandomPerspective(distortion_scale=0.3)

    Simulate 3D rotation:
    - Tangan tegak lurus → tangan miring 30°
    - Training data dari berbagai sudut
    - Model jadi invariant terhadap viewing angle

25. TEST-TIME AUGMENTATION (TTA)
    Technique untuk boost inference accuracy:
    - Apply multiple augmentations saat inference
    - Get predictions dari semua variants
    - Average predictions

    Trade-off: 5× slower, but +1-2% accuracy
    Untuk real-time game: Not recommended
    Untuk critical application: Worth it!

===============================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# ============================================
# KONFIGURASI
# ============================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset" / "cropped"
OUTPUT_DIR = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "training" / "mobilenet_robust_history.json"
PLOT_FILE = BASE_DIR / "training" / "mobilenet_robust_history.png"

OUTPUT_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
NUM_CLASSES = 26

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================
# ROBUST DATA AUGMENTATION
# ============================================

print("\n📚 ROBUST AUGMENTATION PIPELINE:")
print("=" * 60)
print("1. RandomResizedCrop: Scale 0.6-1.0 (simulate distance)")
print("2. RandomPerspective: 30% distortion (simulate angle)")
print("3. RandomAffine: ±30° rotation, 0.8-1.2× scale")
print("4. RandomHorizontalFlip: Left/right hand")
print("5. ColorJitter: Brightness ±30%, contrast ±30%")
print("6. RandomGrayscale: 10% chance (color invariance)")
print("7. GaussianBlur: Simulate out of focus")
print("8. Normalization: ImageNet stats")
print("=" * 60 + "\n")

# Training transforms - AGGRESSIVE AUGMENTATION
train_transform = transforms.Compose(
    [
        # Geometric augmentation
        transforms.RandomResizedCrop(
            224,
            scale=(0.6, 1.0),  # 60-100% crop → simulate distance variation
            ratio=(0.9, 1.1),  # Keep aspect ratio close to square
        ),
        transforms.RandomPerspective(
            distortion_scale=0.3, p=0.5  # 3D rotation simulation  # 50% chance
        ),
        transforms.RandomAffine(
            degrees=30,  # ±30° rotation
            translate=(0.1, 0.1),  # 10% translation
            scale=(0.8, 1.2),  # 80-120% scale
            shear=10,  # ±10° shear
        ),
        transforms.RandomHorizontalFlip(
            p=0.3
        ),  # 30% flip (tidak terlalu sering, BISINDO specific)
        # Color augmentation
        transforms.ColorJitter(
            brightness=0.3,  # ±30% brightness
            contrast=0.3,  # ±30% contrast
            saturation=0.2,  # ±20% saturation
            hue=0.1,  # ±10% hue
        ),
        transforms.RandomGrayscale(p=0.1),  # 10% grayscale (color invariance)
        # Quality augmentation
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize (ImageNet stats)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Random erasing (occlusion simulation)
        transforms.RandomErasing(
            p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)  # 30% chance  # 2-15% area
        ),
    ]
)

# Validation transforms - NO AUGMENTATION
val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ============================================
# DATASET & DATALOADER
# ============================================

print("📂 Loading datasets...")

train_dataset = datasets.ImageFolder(
    root=str(DATA_DIR / "train"), transform=train_transform
)

val_dataset = datasets.ImageFolder(root=str(DATA_DIR / "val"), transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set to 0 for Windows compatibility
    pin_memory=True if device.type == "cuda" else False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,  # Set to 0 for Windows compatibility
    pin_memory=True if device.type == "cuda" else False,
)

print(f"✅ Train samples: {len(train_dataset)}")
print(f"✅ Val samples: {len(val_dataset)}")
print(f"   Batches per epoch: {len(train_loader)}")


# ============================================
# MODEL SETUP
# ============================================

print("\n🧠 Building MobileNetV3-Small model...")

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Linear(576, 1024), nn.Hardswish(), nn.Dropout(0.3), nn.Linear(1024, NUM_CLASSES)
)

model = model.to(device)

print(f"✅ Model loaded: MobileNetV3-Small")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(
    f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M"
)


# ============================================
# TRAINING SETUP
# ============================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler with ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
)

# Training history
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}


# ============================================
# TRAINING FUNCTIONS
# ============================================


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader, desc="Training", unit="batch", leave=False, ncols=100, position=0
    )
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
        )

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(
            loader, desc="Validation", unit="batch", leave=False, ncols=100, position=0
        )
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
            )

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# ============================================
# MAIN TRAINING FUNCTION
# ============================================


def main():
    """
    Main training function.

    📚 KONSEP: Multiprocessing Protection (Windows)

    Windows requires __main__ guard for multiprocessing:
    - DataLoader dengan num_workers > 0 spawn child processes
    - Child processes import main module
    - Without guard → infinite recursion!

    Solution: Wrap training code dalam main() function
    """

    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING - ROBUST AUGMENTATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {NUM_EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Device: {device}")
    print("\n" + "=" * 60 + "\n")

    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = OUTPUT_DIR / "mobilenet_robust_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print epoch results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break

        # Save history after each epoch
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training history saved to: {HISTORY_FILE}")

    # ============================================
    # PLOT TRAINING HISTORY
    # ============================================

    print("\n📊 Plotting training history...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0, 0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss over Epochs")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[0, 1].plot(history["val_acc"], label="Val Acc", marker="s")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Accuracy over Epochs")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning Rate
    axes[1, 0].plot(history["lr"], marker="o", color="green")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True)

    # Train-Val Gap
    gap = np.array(history["train_acc"]) - np.array(history["val_acc"])
    axes[1, 1].plot(gap, marker="o", color="red")
    axes[1, 1].axhline(y=0, color="black", linestyle="--")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy Gap (%)")
    axes[1, 1].set_title("Train-Val Gap (Overfitting Check)")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {PLOT_FILE}")

    plt.close()

    print("\n" + "=" * 60)
    print("🎓 KONSEP YANG DIPELAJARI:")
    print("=" * 60)
    print("✓ Robust data augmentation: Scale, perspective, affine")
    print("✓ Scale invariance: RandomResizedCrop for distance variation")
    print("✓ Perspective transform: 3D rotation simulation")
    print("✓ Color augmentation: Brightness, contrast, saturation")
    print("✓ Occlusion simulation: RandomErasing")
    print("✓ Training monitoring: Loss, accuracy, learning rate")
    print("✓ Overfitting detection: Train-val gap analysis")
    print("=" * 60)

    print("\n🎯 Next Steps:")
    print("1. Evaluate model performance on validation set")
    print("2. Compare with previous model (without robust augmentation)")
    print("3. Test with webcam at various distances/angles")
    print("4. Train multi-modal model (Image + Landmarks)")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
