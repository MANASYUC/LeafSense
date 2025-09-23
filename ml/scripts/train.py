import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.training_config import TrainingConfig


class PlantDiseaseTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.validate()
        
        # Set device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
            
        print(f"Using device: {self.device}")
        print(f"Using {self.config.data_fraction*100:.0f}% of training data for faster training")
        
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
        """Build training and validation transforms"""
        if self.config.use_augmentation:
            train_tf = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomRotation(self.config.rotation_degrees),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
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
        
        # Create data subsets for faster training
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
        """Build and initialize the model"""
        if self.config.model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif self.config.model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        elif self.config.model_name == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b1(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        elif self.config.model_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b2(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        elif self.config.model_name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b3(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        elif self.config.model_name == "efficientnet_b4":
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b4(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Model {self.config.model_name} initialized with {self.num_classes} classes")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'class_to_idx': self.class_to_idx,
            'num_classes': self.num_classes,
        }
        
        # Save best model
        if is_best:
            best_path = self.models_dir / f"{self.config.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
            
            # Save class mapping
            class_path = self.models_dir / f"{self.config.model_name}_classes.json"
            with open(class_path, 'w') as f:
                json.dump(self.class_to_idx, f, indent=2)
        
        # Save last model if requested
        if self.config.save_last:
            last_path = self.models_dir / f"{self.config.model_name}_last.pt"
            torch.save(checkpoint, last_path)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate scheduling
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
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train plant disease classification model")
    parser.add_argument("--model", type=str, default="resnet50", 
                       choices=["resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"], 
                       help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data-fraction", type=float, default=0.15, 
                       help="Fraction of data to use (0.15 = 15%% of data, good for laptops)")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_fraction=args.data_fraction,
        device=args.device
    )
    
    # Create trainer and start training
    trainer = PlantDiseaseTrainer(config)
    trainer.create_dataloaders()
    trainer.build_model()
    trainer.train()


if __name__ == "__main__":
    main()
