import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.training_config import TrainingConfig


class ModelValidator:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        
        # Load checkpoint
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = TrainingConfig(**self.checkpoint['config'])
        self.class_to_idx = self.checkpoint['class_to_idx']
        self.num_classes = self.checkpoint['num_classes']
        
        # Create reverse mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Setup paths
        self.setup_paths()
        
        # Initialize model
        self.model = None
        self.val_loader = None
        self.criterion = None
        
        print(f"Loaded model from: {self.model_path}")
        print(f"Model: {self.config.model_name}")
        print(f"Classes: {self.num_classes}")
        print(f"Device: {self.device}")
    
    def setup_paths(self):
        """Setup data paths"""
        self.data_root = Path(__file__).parent.parent.parent / "data"
        self.valid_dir = self.data_root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "valid"
        
        if not self.valid_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {self.valid_dir}")
    
    def build_model(self):
        """Build and load the model"""
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
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        # Load trained weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        print("Model loaded successfully")
    
    def create_dataloader(self):
        """Create validation dataloader"""
        val_tf = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val_dataset = datasets.ImageFolder(self.valid_dir, transform=val_tf)
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=False  # No need for validation
        )
        
        print(f"Validation samples: {len(val_dataset)}")
    
    def validate(self) -> Tuple[float, float, List[int], List[int], List[float]]:
        """Run validation and return metrics"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_predictions, all_targets, all_confidences
    
    def generate_report(self, predictions: List[int], targets: List[int], confidences: List[float]):
        """Generate detailed validation report"""
        # Convert indices to class names
        pred_classes = [self.idx_to_class[p] for p in predictions]
        true_classes = [self.idx_to_class[t] for t in targets]
        
        # Classification report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(true_classes, pred_classes, target_names=list(self.class_to_idx.keys())))
        
        # Confusion matrix
        print("\n" + "="*50)
        print("CONFUSION MATRIX")
        print("="*50)
        cm = confusion_matrix(targets, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.class_to_idx.keys()),
                   yticklabels=list(self.class_to_idx.keys()))
        plt.title(f'Confusion Matrix - {self.config.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(__file__).parent.parent / "models" / f"{self.config.model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {plot_path}")
        
        # Per-class accuracy
        print("\n" + "="*50)
        print("PER-CLASS ACCURACY")
        print("="*50)
        for i in range(self.num_classes):
            class_mask = np.array(targets) == i
            if class_mask.sum() > 0:
                class_acc = (np.array(predictions)[class_mask] == i).mean() * 100
                class_name = self.idx_to_class[i]
                print(f"{class_name:30s}: {class_acc:6.2f}% ({class_mask.sum():3d} samples)")
        
        # Confidence analysis
        print("\n" + "="*50)
        print("CONFIDENCE ANALYSIS")
        print("="*50)
        confidences = np.array(confidences)
        correct_mask = np.array(predictions) == np.array(targets)
        
        print(f"Average confidence (correct): {confidences[correct_mask].mean():.3f}")
        print(f"Average confidence (incorrect): {confidences[~correct_mask].mean():.3f}")
        print(f"Confidence std (correct): {confidences[correct_mask].std():.3f}")
        print(f"Confidence std (incorrect): {confidences[~correct_mask].std():.3f}")
    
    def run_validation(self):
        """Run complete validation"""
        print("Building model...")
        self.build_model()
        
        print("Creating dataloader...")
        self.create_dataloader()
        
        print("Running validation...")
        val_loss, val_acc, predictions, targets, confidences = self.validate()
        
        print(f"\nValidation Results:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_acc:.2f}%")
        
        # Generate detailed report
        self.generate_report(predictions, targets, confidences)


def main():
    parser = argparse.ArgumentParser(description="Validate trained plant disease model")
    parser.add_argument("model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Run validation
    validator = ModelValidator(args.model_path, args.device)
    validator.run_validation()


if __name__ == "__main__":
    main()
