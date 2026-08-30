"""
Parking Spot CNN Classifier Training Pipeline
Trains a CNN model to classify parking spots as 'empty' (0) or 'occupied' (1).
Optimized for macOS with Apple Silicon (MPS / CPU).
"""

import os
import sys
import glob
import time
import pickle
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU Acceleration (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("ℹ️ Using CPU device")

# Image dimensions
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

class ParkingSpotDataset(Dataset):
    """Custom Dataset for loading parking spot image patches."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class ParkingCNN(nn.Module):
    """Lightweight & high-accuracy CNN architecture for parking slot occupancy detection."""
    def __init__(self):
        super(ParkingCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2)  # 2 classes: 0 = empty, 1 = occupied
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def collect_images_and_labels(base_dirs):
    """Scans provided directories for 'empty' and 'occupied' image patches."""
    empty_paths = []
    occupied_paths = []

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    for bdir in base_dirs:
        if not os.path.exists(bdir):
            continue

        # Check subdirectories
        for root, dirs, files in os.walk(bdir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    full_path = os.path.join(root, f)
                    if 'empty' in root.lower() or 'empty' in f.lower():
                        empty_paths.append(full_path)
                    elif 'occupied' in root.lower() or 'occupied' in f.lower() or 'car' in root.lower():
                        occupied_paths.append(full_path)

    # Remove duplicates
    empty_paths = sorted(list(set(empty_paths)))
    occupied_paths = sorted(list(set(occupied_paths)))

    paths = empty_paths + occupied_paths
    labels = [0] * len(empty_paths) + [1] * len(occupied_paths)

    return paths, labels, len(empty_paths), len(occupied_paths)

def train_model(train_dir="/Users/abhishekkumar/Desktop/train_data/train",
                test_dir="/Users/abhishekkumar/Desktop/train_data/test",
                output_dir="/Users/abhishekkumar/Desktop/parking_space_detection/models"):
    """Main training routine."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("🅿️ PARKING SPACE CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)

    # Search paths for train
    train_dirs = [
        train_dir,
        "/Users/abhishekkumar/Desktop/train_data/train",
        "/Users/abhishekkumar/Desktop/train",
        "/Users/abhishekkumar/Desktop/empty",
        "/Users/abhishekkumar/Desktop/occupied"
    ]
    
    # Search paths for test
    test_dirs = [
        test_dir,
        "/Users/abhishekkumar/Desktop/train_data/test"
    ]

    print(f"🔍 Loading training dataset from: {train_dir}")
    train_paths, train_labels, n_train_empty, n_train_occ = collect_images_and_labels([train_dir])
    
    if len(train_paths) == 0:
        print("⚠️ No images found in primary train_dir, scanning Desktop fallback directories...")
        train_paths, train_labels, n_train_empty, n_train_occ = collect_images_and_labels(train_dirs)

    print(f"📊 Training samples: {len(train_paths)} (Empty: {n_train_empty}, Occupied: {n_train_occ})")

    print(f"🔍 Loading testing dataset from: {test_dir}")
    test_paths, test_labels, n_test_empty, n_test_occ = collect_images_and_labels([test_dir])

    if len(test_paths) == 0:
        print("⚠️ Test split empty or not found. Splitting training data (80/20)...")
        from sklearn.model_selection import train_test_split
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
    else:
        print(f"📊 Testing samples: {len(test_paths)} (Empty: {n_test_empty}, Occupied: {n_test_occ})")

    # Data transforms & augmentations
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ParkingSpotDataset(train_paths, train_labels, transform=train_transforms)
    test_dataset = ParkingSpotDataset(test_paths, test_labels, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, optimizer
    model = ParkingCNN().to(device)
    
    # Class weights to handle class imbalance (occupied typically has more samples)
    class_counts = [max(n_train_empty, 1), max(n_train_occ, 1)]
    weights = [sum(class_counts) / (2.0 * c) for c in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "model_final.pth")
    root_model_path = "/Users/abhishekkumar/Desktop/parking_space_detection/model_final.pth"
    
    print("\n⚡ Starting Training Loop...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = (val_correct / val_total) * 100.0

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        if val_epoch_acc >= best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), root_model_path)

        if epoch % 2 == 0 or epoch == NUM_EPOCHS:
            print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
                  f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
                  f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.2f}s! Best Validation Accuracy: {best_val_acc:.2f}%")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("📈 FINAL MODEL EVALUATION METRICS ON TEST SET:")
    print("=" * 60)
    target_names = ["Empty", "Occupied"]
    report = classification_report(all_targets, all_preds, target_names=target_names)
    print(report)

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)

    # Plot training curves & confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Loss & Accuracy Curves
    axes[0].plot(history['train_acc'], label='Train Accuracy', color='#3b82f6', lw=2)
    axes[0].plot(history['val_acc'], label='Val Accuracy', color='#10b981', lw=2)
    axes[0].set_title('Model Accuracy vs Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # 2. Confusion Matrix Heatmap
    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=axes[1])
    tick_marks = np.arange(len(target_names))
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(target_names)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(target_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=14, fontweight='bold')

    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Training curves and confusion matrix saved to: {plot_path}")

    # Also save model metadata
    metadata = {
        'img_size': IMG_SIZE,
        'classes': ['empty', 'occupied'],
        'best_val_acc': best_val_acc,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(output_dir, "model_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    with open("/Users/abhishekkumar/Desktop/parking_space_detection/model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"💾 Model artifact saved to: {best_model_path} & {root_model_path}")
    print("=" * 60)
    return best_model_path

if __name__ == "__main__":
    train_model()
