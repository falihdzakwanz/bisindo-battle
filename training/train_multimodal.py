"""
BISINDO Multi-Modal Training: Image + Hand Landmarks

📚 KONSEP MULTI-MODAL DEEP LEARNING:
===============================================

26. MULTI-MODAL ARCHITECTURE
    Multi-Modal = Model yang process multiple types of data

    Architecture Pattern:
    ┌─────────────────┐         ┌──────────────────┐
    │  Image (224x224)│         │ Landmarks (21x3) │
    └────────┬────────┘         └────────┬─────────┘
             │                           │
             ▼                           ▼
    ┌─────────────────┐         ┌──────────────────┐
    │  CNN Backbone   │         │   MLP Encoder    │
    │  (MobileNetV3)  │         │  (2-3 layers)    │
    └────────┬────────┘         └────────┬─────────┘
             │                           │
             │  Features (576)           │  Features (128)
             │                           │
             └───────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Concatenate (704)│
              └─────────┬────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Fusion Layer    │
              │   (512 neurons)  │
              └─────────┬────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Output (26)     │
              │   A-Z classes    │
              └──────────────────┘

    Kenapa effective?
    - CNN capture texture, color, context (visual)
    - MLP capture geometry, angles, pose (structural)
    - Fusion layer learn optimal combination
    - If one modality noisy → other modality compensate!

27. EARLY FUSION vs LATE FUSION

    A. EARLY FUSION (yang kita pakai)
       - Combine features di middle layer
       - Allow interaction between modalities
       - More flexible, better accuracy

    B. LATE FUSION
       - Each modality → separate prediction
       - Combine predictions (voting/averaging)
       - Simpler, but less interaction

    C. HYBRID FUSION
       - Multiple fusion points
       - Complex, but best performance
       - Overkill untuk BISINDO

28. LANDMARK MLP ENCODER
    Input: 63 features (21 landmarks × 3 coords)

    Architecture:
    - Layer 1: 63 → 256 (expand features)
    - Layer 2: 256 → 128 (compress to match CNN)
    - Activation: ReLU
    - Normalization: BatchNorm1d
    - Regularization: Dropout(0.3)

    Why MLP for landmarks?
    - Landmarks already geometric (no spatial structure)
    - MLP enough for coordinate processing
    - CNN would be overkill (landmarks ≠ images)

29. CUSTOM DATASET FOR MULTI-MODAL
    PyTorch Dataset requirements:
    - __len__(): Return number of samples
    - __getitem__(idx): Return (image, landmarks, label)

    Challenge: Handle missing landmarks
    - Some images failed landmark extraction
    - Solution: Skip those samples OR use zero-padding
    - We choose: Skip (cleaner, no fake data)

30. FUSION STRATEGIES
    Simple Concatenation:
    - features = torch.cat([image_features, landmark_features], dim=1)

    Weighted Fusion:
    - features = α * image_features + β * landmark_features
    - α, β learned during training

    Attention Fusion:
    - Attention weights untuk prioritize modalities
    - Complex, but powerful

    For BISINDO: Concatenation sufficient (simple & effective)

===============================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================
# KONFIGURASI
# ============================================

BASE_DIR = Path(__file__).parent.parent
CROPPED_DIR = BASE_DIR / "dataset" / "cropped"
LANDMARKS_DIR = BASE_DIR / "dataset" / "landmarks"
OUTPUT_DIR = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "training" / "multimodal_history.json"
PLOT_FILE = BASE_DIR / "training" / "multimodal_history.png"

OUTPUT_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
NUM_CLASSES = 26
LANDMARK_DIM = 63  # 21 points × 3 coordinates

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ============================================
# MULTI-MODAL DATASET
# ============================================


class MultiModalBISINDODataset(Dataset):
    """
    Custom Dataset untuk Multi-Modal Learning.

    📚 KONSEP: Multi-Modal Data Loading

    Return tuple: (image, landmarks, label)
    - image: Tensor (3, 224, 224)
    - landmarks: Tensor (63,) - normalized coordinates
    - label: int (0-25 untuk A-Z)

    Handle missing landmarks:
    - Filter out samples yang landmark extraction failed
    - Only keep samples dengan both image & landmarks available
    """

    def __init__(self, image_dir, landmark_dir, transform=None):
        """
        Args:
            image_dir: Path ke cropped images
            landmark_dir: Path ke landmarks (.npy files)
            transform: Image transformations
        """
        self.image_dir = Path(image_dir)
        self.landmark_dir = Path(landmark_dir)
        self.transform = transform

        # Get all valid samples (images + landmarks)
        self.samples = []
        self.class_to_idx = {chr(65 + i): i for i in range(26)}  # A-Z → 0-25

        print(f"\n📂 Loading multi-modal dataset from:")
        print(f"   Images: {image_dir}")
        print(f"   Landmarks: {landmark_dir}")

        # Iterate classes
        for class_name in sorted(self.class_to_idx.keys()):
            class_image_dir = self.image_dir / class_name
            class_landmark_dir = self.landmark_dir / class_name

            if not class_image_dir.exists() or not class_landmark_dir.exists():
                continue

            # Get images with corresponding landmarks
            image_files = list(class_image_dir.glob("*.jpg"))
            valid_count = 0

            for image_path in image_files:
                landmark_path = class_landmark_dir / (image_path.stem + ".npy")

                # Only add if landmark exists
                if landmark_path.exists():
                    self.samples.append(
                        {
                            "image_path": image_path,
                            "landmark_path": landmark_path,
                            "label": self.class_to_idx[class_name],
                        }
                    )
                    valid_count += 1

            print(
                f"   Class {class_name}: {valid_count}/{len(image_files)} valid samples"
            )

        print(f"\n✅ Total valid samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found! Check landmark extraction.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load landmarks
        landmarks = np.load(sample["landmark_path"])
        landmarks = torch.from_numpy(landmarks).float()

        # Label
        label = sample["label"]

        return image, landmarks, label


# ============================================
# DATA TRANSFORMS
# ============================================

# Training transforms - Robust augmentation
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomAffine(
            degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
        ),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ]
)

# Validation transforms
val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ============================================
# DATASETS & DATALOADERS
# ============================================

print("\n" + "=" * 60)
print("📚 LOADING MULTI-MODAL DATASETS")
print("=" * 60)

train_dataset = MultiModalBISINDODataset(
    image_dir=CROPPED_DIR / "train",
    landmark_dir=LANDMARKS_DIR / "train",
    transform=train_transform,
)

val_dataset = MultiModalBISINDODataset(
    image_dir=CROPPED_DIR / "val",
    landmark_dir=LANDMARKS_DIR / "val",
    transform=val_transform,
)

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

print(f"\n✅ Train samples: {len(train_dataset)}")
print(f"✅ Val samples: {len(val_dataset)}")


# ============================================
# MULTI-MODAL MODEL
# ============================================


class MultiModalBISINDO(nn.Module):
    """
    Multi-Modal BISINDO Recognition Model.

    📚 KONSEP: Multi-Modal Architecture

    Two branches:
    1. Image branch: MobileNetV3 (pretrained)
    2. Landmark branch: MLP encoder

    Fusion: Concatenate features → Classification head
    """

    def __init__(self, num_classes=26, landmark_dim=63):
        super(MultiModalBISINDO, self).__init__()

        # Image branch: MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Extract feature extractor (remove classifier)
        self.image_encoder = mobilenet.features
        self.image_pool = mobilenet.avgpool

        # Image feature dimension: 576
        self.image_feature_dim = 576

        # Landmark branch: MLP
        self.landmark_encoder = nn.Sequential(
            nn.Linear(landmark_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.landmark_feature_dim = 128

        # Fusion layer
        combined_dim = (
            self.image_feature_dim + self.landmark_feature_dim
        )  # 576 + 128 = 704

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, landmarks):
        """
        Forward pass dengan dual inputs.

        Args:
            image: Tensor (batch, 3, 224, 224)
            landmarks: Tensor (batch, 63)

        Returns:
            logits: Tensor (batch, num_classes)
        """
        # Image branch
        img_features = self.image_encoder(image)
        img_features = self.image_pool(img_features)
        img_features = torch.flatten(img_features, 1)  # (batch, 576)

        # Landmark branch
        lm_features = self.landmark_encoder(landmarks)  # (batch, 128)

        # Concatenate features
        combined = torch.cat([img_features, lm_features], dim=1)  # (batch, 704)

        # Classification
        output = self.fusion(combined)

        return output


# ============================================
# MODEL INITIALIZATION
# ============================================

print("\n" + "=" * 60)
print("🧠 BUILDING MULTI-MODAL MODEL")
print("=" * 60)
print("\nArchitecture:")
print("  Image Branch: MobileNetV3-Small → 576 features")
print("  Landmark Branch: MLP (63→256→128) → 128 features")
print("  Fusion: Concatenate (704) → 512 → 26 classes")

model = MultiModalBISINDO(num_classes=NUM_CLASSES, landmark_dim=LANDMARK_DIM)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✅ Model created!")
print(f"   Total parameters: {total_params / 1e6:.2f}M")
print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")


# ============================================
# TRAINING SETUP
# ============================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}


# ============================================
# TRAINING FUNCTIONS
# ============================================


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader, desc="Training", unit="batch", leave=False, ncols=100, position=0
    )
    for images, landmarks, labels in pbar:
        images = images.to(device)
        landmarks = landmarks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, landmarks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
        )

    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(
            loader, desc="Validation", unit="batch", leave=False, ncols=100, position=0
        )
        for images, landmarks, labels in pbar:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": running_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
            )

    return running_loss / len(loader), 100.0 * correct / total


# ============================================
# MAIN TRAINING FUNCTION
# ============================================


def main():
    """Main training function with multiprocessing protection."""

    print("\n" + "=" * 60)
    print("🚀 STARTING MULTI-MODAL TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Modalities: Image (224×224×3) + Landmarks (21×3)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {NUM_EPOCHS}")
    print(f"  Device: {device}")
    print("\n" + "=" * 60 + "\n")

    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = OUTPUT_DIR / "multimodal_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ MULTI-MODAL TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")

    # Plot history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train", marker="o")
    axes[0].plot(history["val_loss"], label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Multi-Modal Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train", marker="o")
    axes[1].plot(history["val_acc"], label="Val", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Multi-Modal Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to: {PLOT_FILE}")

    print("\n🎓 MULTI-MODAL CONCEPTS LEARNED:")
    print("=" * 60)
    print("✓ Multi-modal architecture: Dual input streams")
    print("✓ Feature fusion: Concatenation strategy")
    print("✓ Landmark encoding: MLP for geometric features")
    print("✓ Custom dataset: Handle multiple data types")
    print("✓ Complementary information: Visual + geometric")
    print("=" * 60)


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
