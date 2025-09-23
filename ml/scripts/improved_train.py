import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.improved_training_config import ImprovedTrainingConfig


class ImprovedPlantDiseaseTrainer:
    def __init__(self, config: ImprovedTrainingConfig):
        self.config = config
        self.config.validate()
        
        # Set device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
            
        print(f"Using device: {self.device}")
        print(f"Using {self.config.data_fraction*100:.0f}% of training data")
        
        # Setup paths
        self.setup_paths()
        
        # Initialize model and data
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
    def setup_paths(self):
        """Setup data and model paths"""
        # Data paths
        self.data_root = Path(__file__).parent.parent.parent / "data"
        self.train_dir = self.data_root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train"
        self.valid_dir = self.data_root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "valid"
        
        # Model paths
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Check if data directories exist
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        if not self.valid_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {self.valid_dir}")
            
        print(f"Training data: {self.train_dir}")
        print(f"Validation data: {self.valid_dir}")
        print(f"Models will be saved to: {self.models_dir}")
    
    def build_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        """Build enhanced training and validation transforms"""
        if self.config.use_augmentation:
            augmentation_transforms = [
                transforms.Resize((self.config.image_size + 32, self.config.image_size + 32)),
            ]
            
            if self.config.use_random_crop:
                augmentation_transforms.append(
                    transforms.RandomCrop((self.config.image_size, self.config.image_size))
                )
            else:
                augmentation_transforms.append(
                    transforms.Resize((self.config.image_size, self.config.image_size))
                )
            
            augmentation_transforms.extend([
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomRotation(self.config.rotation_degrees),
            ])
            
            if self.config.color_jitter:
                augmentation_transforms.append(
                    transforms.ColorJitter(
                        brightness=self.config.color_jitter_brightness,
                        contrast=self.config.color_jitter_contrast,
                        saturation=self.config.color_jitter_saturation,
                        hue=self.config.color_jitter_hue
                    )
                )
            
            augmentation_transforms.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            if self.config.use_random_erasing:
                augmentation_transforms.append(
                    transforms.RandomErasing(p=self.config.random_erasing_prob)
                )
            
            train_tf = transforms.Compose(augmentation_transforms)
        else:
            train_tf = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        val_tf = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return train_tf, val_tf
    
    def create_dataloaders(self):
        """Create training and validation dataloaders with data subset"""
        train_tf, val_tf = self.build_transforms()
        
        # Create full datasets
        full_train_dataset = datasets.ImageFolder(self.train_dir, transform=train_tf)
        full_val_dataset = datasets.ImageFolder(self.valid_dir, transform=val_tf)
        
        # Create data subsets for training
        train_size = int(len(full_train_dataset) * self.config.data_fraction)
        val_size = int(len(full_val_dataset) * self.config.data_fraction)
        
        # Randomly sample indices
        train_indices = random.sample(range(len(full_train_dataset)), train_size)
        val_indices = random.sample(range(len(full_val_dataset)), val_size)
        
        # Create subset datasets
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )
        
        # Save class mapping from full dataset
        self.class_to_idx = full_train_dataset.class_to_idx
        self.num_classes = len(full_train_dataset.classes)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Full training samples: {len(full_train_dataset)}")
        print(f"Subset training samples: {len(train_dataset)} ({self.config.data_fraction*100:.0f}%)")
        print(f"Full validation samples: {len(full_val_dataset)}")
        print(f"Subset validation samples: {len(val_dataset)} ({self.config.data_fraction*100:.0f}%)")
    
    def build_model(self):
        """Build and initialize the improved model"""
        # Get the appropriate weights for the model
        weights_map = {
            "efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
            "efficientnet_b1": EfficientNet_B1_Weights.IMAGENET1K_V1,
            "efficientnet_b2": EfficientNet_B2_Weights.IMAGENET1K_V1,
            "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
            "efficientnet_b4": EfficientNet_B4_Weights.IMAGENET1K_V1,
        }
        
        if self.config.model_name in weights_map:
            weights = weights_map[self.config.model_name]
            self.model = getattr(models, self.config.model_name)(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        if self.config.use_cosine_annealing:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.config.cosine_restart_epochs,
                T_mult=2
            )
        else:
            self.scheduler = None
        
        # Initialize loss function with label smoothing
        if self.config.use_label_smoothing:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing_factor)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        print(f"Model {self.config.model_name} initialized with {self.num_classes} classes")
        print(f"Using label smoothing: {self.config.use_label_smoothing}")
        print(f"Using cosine annealing: {self.config.use_cosine_annealing}")
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Compute mixup loss"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def cutmix_data(self, x, y, alpha=1.0):
        """Apply cutmix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box for cutmix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with enhanced techniques"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Apply mixup or cutmix
            if self.config.use_mixup and np.random.random() < 0.5:
                images, targets_a, targets_b, lam = self.mixup_data(images, targets, self.config.mixup_alpha)
            elif self.config.use_cutmix and np.random.random() < 0.5:
                images, targets_a, targets_b, lam = self.cutmix_data(images, targets, self.config.cutmix_alpha)
            else:
                targets_a, targets_b, lam = targets, targets, 1
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            if lam == 1:
                loss = self.criterion(outputs, targets)
            else:
                loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'acc': val_acc}
    
    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.__dict__,
            'class_to_idx': self.class_to_idx,
            'num_classes': self.num_classes,
        }
        
        # Save best model
        if is_best:
            best_path = self.models_dir / f"{self.config.model_name}_improved_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
            
            # Save class mapping
            class_path = self.models_dir / f"{self.config.model_name}_improved_classes.json"
            with open(class_path, 'w') as f:
                json.dump(self.class_to_idx, f, indent=2)
        
        # Save last model if requested
        if self.config.save_last:
            last_path = self.models_dir / f"{self.config.model_name}_improved_last.pt"
            torch.save(checkpoint, last_path)
    
    def train(self):
        """Main improved training loop"""
        print(f"Starting improved training for {self.config.epochs} epochs...")
        print(f"Using mixup: {self.config.use_mixup}")
        print(f"Using cutmix: {self.config.use_cutmix}")
        print(f"Using label smoothing: {self.config.use_label_smoothing}")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Print metrics
            print(f"Epoch {epoch}/{self.config.epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%")
            
            # Save best model
            is_best = val_metrics['acc'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['acc']
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_model(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        print(f"Improved training completed. Best validation accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train improved plant disease classification model")
    parser.add_argument("--model", type=str, default="efficientnet_b3", 
                       choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"], 
                       help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-fraction", type=float, default=0.6, 
                       help="Fraction of data to use (0.6 = 60% of data)")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Create improved config
    config = ImprovedTrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_fraction=args.data_fraction,
        device=args.device
    )
    
    # Create trainer and start training
    trainer = ImprovedPlantDiseaseTrainer(config)
    trainer.create_dataloaders()
    trainer.build_model()
    trainer.train()


if __name__ == "__main__":
    main()
