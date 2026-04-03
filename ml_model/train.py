"""
AI vs Real Image Detection - PyTorch Model Training
Compatible with Python 3.12, 3.13, 3.14
"""

import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

def print_header():
    """Print formatted header"""
    print("\n" + "="*60)
    print("  AI IMAGE FORENSICS - MODEL TRAINING")
    print("="*60)
    print(f"   Device: {device}")
    print()

def print_step(step_num, total_steps, message):
    """Print step header"""
    print(f"[Step {step_num}/{total_steps}] {message}")

def print_stat(label, value, indent=3):
    """Print formatted stat"""
    print(" "*indent + f"{label}: {value}")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 13       # Reduced to get ~300 batches per epoch
EPOCHS = 5          # Increased for better accuracy
LEARNING_RATE = 0.0001  # Lower for better convergence
DATASET_PATH = "../datasets/train"
MODEL_SAVE_PATH = "ai_real_classifier.pth"
CLASS_INDICES_PATH = "class_indices.pkl"
QUICK_MODE = False    # Use full dataset for better accuracy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ImageDataset(Dataset):
    """Custom dataset for loading images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CNNClassifier(nn.Module):
    """CNN model using full ResNet50 with all layers trainable"""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Use pretrained ResNet50 (more powerful than ResNet18)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # UNFREEZE all layers for full training (better accuracy)
        # All parameters will be trainable by default
        
        # Replace classifier with custom head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)


def load_dataset(dataset_path, max_images_per_class=500):
    """Load dataset with optional limit for fast training"""
    image_paths = []
    labels = []
    class_names = []
    
    # Get class names (FAKE, REAL)
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
            class_idx = len(class_names) - 1
            
            # Get images (limited for quick mode)
            images_in_class = []
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    img_path = os.path.join(class_dir, img_name)
                    images_in_class.append(img_path)
            
            # Limit images per class for fast training
            if QUICK_MODE and len(images_in_class) > max_images_per_class:
                import random
                random.seed(42)
                images_in_class = random.sample(images_in_class, max_images_per_class)
                print(f"  {class_name}: Using {max_images_per_class}/{len(os.listdir(class_dir))} images (quick mode)")
            
            for img_path in images_in_class:
                image_paths.append(img_path)
                labels.append(class_idx)
    
    print(f"Found {len(image_paths)} images in {len(class_names)} classes: {class_names}")
    
    # Save class indices
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    with open(CLASS_INDICES_PATH, 'wb') as f:
        pickle.dump(class_indices, f)
    print(f"Class indices: {class_indices}")
    
    return image_paths, labels, class_names


def get_transforms(is_training=True):
    """Get image transforms for training/validation with enhanced augmentation"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def train_model():
    """Train the model with formatted output"""
    print_header()
    
    # Step 1: Load dataset
    print_step(1, 5, "Loading dataset...")
    image_paths, labels, class_names = load_dataset(DATASET_PATH)
    
    # Split into train/validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print_stat("Training samples", len(train_paths))
    print_stat("Validation samples", len(val_paths))
    class_mapping = ", ".join([f"{name}={idx}" for name, idx in {name: idx for idx, name in enumerate(class_names)}.items()])
    print_stat("Class mapping", class_mapping)
    print()
    
    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, transform=get_transforms(True))
    val_dataset = ImageDataset(val_paths, val_labels, transform=get_transforms(False))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Step 2: Build model
    print_step(2, 5, "Building ResNet50 model...")
    model = CNNClassifier().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_stat("Total parameters", f"{total_params:,}")
    print_stat("Trainable parameters", f"{trainable_params:,}")
    
    # Class weights
    weight_real = len(train_labels) / (2 * sum(1 for l in train_labels if l == 0))
    weight_fake = len(train_labels) / (2 * sum(1 for l in train_labels if l == 1))
    print_stat("Class weights", f"REAL={weight_real:.2f}, FAKE={weight_fake:.2f}")
    print()
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2)
    
    # Step 3: Training
    print_step(3, 5, f"Training for up to {EPOCHS} epochs...")
    batches_per_epoch = len(train_loader)
    print_stat("Batches per epoch", batches_per_epoch)
    print("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels_batch) in enumerate(train_loader):
            images = images.to(device)
            labels_batch = labels_batch.float().to(device)
            
            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()
            
            # Print batch progress (single line update)
            if batch_idx % 5 == 0 or batch_idx == batches_per_epoch - 1:
                progress = (batch_idx / batches_per_epoch) * 100
                batch_acc = 100 * train_correct / train_total if train_total > 0 else 0
                print(f"\r   Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx}/{batches_per_epoch} ({progress:.1f}%) - Loss: {loss.item():.4f} - Acc: {batch_acc:.1f}%", end='', flush=True)
        
        print()  # New line after epoch completes
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.float().to(device)
                
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"   Epoch {epoch+1}/{EPOCHS} Summary - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'img_size': IMG_SIZE
            }, MODEL_SAVE_PATH)
            print(f"   -> Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model


def evaluate_on_test_set():
    """Evaluate the trained model on the test set"""
    test_path = "../datasets/test"
    
    if not os.path.exists(test_path):
        print(f"\nTest path {test_path} does not exist. Skipping test evaluation.")
        return
    
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    # Load model
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model = CNNClassifier().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_images, test_labels, class_names = load_dataset(test_path)
    test_dataset = ImageDataset(test_images, test_labels, transform=get_transforms(False))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Predict
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).int()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    # Train model
    model = train_model()
    
    # Evaluate on test set
    evaluate_on_test_set()
    
    print("\n" + "=" * 60)
    print("All Done! Model saved to:", MODEL_SAVE_PATH)
    print("=" * 60)
