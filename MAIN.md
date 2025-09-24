# Plant Disease Detection - Training Manual (MANI)

## Setup and Prerequisites

Before starting any training, ensure you have the proper environment set up:

### 1. Install Python Dependencies
```bash
cd ml
pip install -r requirements.txt
```

### 2. Verify Data Structure
Ensure your data is organized as follows:
```
data/
└── New Plant Diseases Dataset(Augmented)/
    └── New Plant Diseases Dataset(Augmented)/
        ├── train/          # Training images
        └── valid/          # Validation images
```

### 3. Check GPU Availability (Optional)
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
```

---

## Available Models

The following models are supported for plant disease classification:

### ResNet Models
- **resnet50** - Classic CNN architecture, good balance of speed and accuracy

### EfficientNet Models (Recommended)
- **efficientnet_b0** - Fastest, good for quick experiments
- **efficientnet_b1** - Slightly larger, better accuracy
- **efficientnet_b2** - Medium size, good accuracy/speed balance
- **efficientnet_b3** - Larger model, higher accuracy (recommended)
- **efficientnet_b4** - Largest, highest accuracy but slower training

### Model Characteristics
- **38 classes** total (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- **Pre-trained weights** from ImageNet
- **Transfer learning** approach for faster convergence

---

## General Training Script

### Basic Training Command
```bash
cd ml/scripts
python train.py --model efficientnet_b3 --epochs 15 --batch-size 32 --lr 3e-4 --data-fraction 0.15
```

### Fast Training (for quick experiments)
```bash
python fast_train.py --epochs 5 --batch-size 64 --lr 5e-4 --data-fraction 0.3
```

### Improved Training (for better accuracy)
```bash
python improved_train.py --model efficientnet_b3 --epochs 30 --batch-size 32 --lr 1e-4 --data-fraction 0.6
```

### Enhanced Training (for maximum accuracy)
```bash
python enhanced_train.py --model efficientnet_b3 --epochs 25 --batch-size 32 --lr 1e-4 --data-fraction 0.6
```

---

## Training Parameters and Flags

### Model Selection
- **--model** `[resnet50, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4]`
  - Choose the neural network architecture to train

### Training Hyperparameters
- **--epochs** `[1-100]` (default: varies by script)
  - Number of complete passes through the training data
- **--batch-size** `[8-128]` (default: 32)
  - Number of samples processed in one forward/backward pass
- **--lr** `[1e-6 to 1e-2]` (default: varies by script)
  - Learning rate for the optimizer (lower = more stable, higher = faster convergence)
- **--data-fraction** `[0.01-1.0]` (default: varies by script)
  - Fraction of total dataset to use (0.15 = 15% of data, good for laptops)

### Device Configuration
- **--device** `[auto, cuda, cpu]` (default: auto)
  - Choose computing device (auto detects GPU if available)

### Data Processing
- **--image-size** `[224, 256, 300, 384]` (default: varies by script)
  - Input image resolution (higher = more detail but slower)
- **--num-workers** `[1-16]` (default: 4)
  - Number of parallel data loading processes
- **--pin-memory** `[True, False]` (default: True)
  - Pin memory for faster GPU transfer





---

## Training Scripts Overview

### 1. `train.py` - Standard Training
- **Purpose**: General-purpose training with basic augmentation
- **Best for**: Getting started, baseline models
- **Default settings**: 15 epochs, 15% data, basic augmentation

### 2. `fast_train.py` - Quick Training
- **Purpose**: Fast experimentation and testing
- **Best for**: Quick iterations, testing changes
- **Default settings**: 5 epochs, 30% data, minimal augmentation

### 3. `improved_train.py` - Enhanced Training
- **Purpose**: Better accuracy with advanced techniques
- **Best for**: Production models, accuracy-focused training
- **Default settings**: 30 epochs, 60% data, advanced augmentation

### 4. `enhanced_train.py` - Maximum Accuracy
- **Purpose**: Highest possible accuracy
- **Best for**: Final models, competition submissions
- **Default settings**: 25 epochs, 60% data, all advanced techniques

---

## Example Training Commands

### Quick Test (5 minutes)
```bash
python fast_train.py --epochs 3 --data-fraction 0.1
```

### Standard Training (30 minutes)
```bash
python train.py --model efficientnet_b0 --epochs 10 --data-fraction 0.3
```

### Production Training (2-4 hours)
```bash
python improved_train.py --model efficientnet_b3 --epochs 25 --data-fraction 0.8
```

### Maximum Accuracy (4-8 hours)
```bash
python enhanced_train.py --model efficientnet_b4 --epochs 40 --data-fraction 1.0
```

---

## Monitoring Training

### Real-time Metrics
- **Loss**: Should decrease over time
- **Accuracy**: Should increase over time
- **Validation**: Should track training closely (watch for overfitting)

### Early Stopping
- Training stops if validation accuracy doesn't improve for `early-stopping-patience` epochs
- Prevents overfitting and saves time

### Model Checkpoints
- Best model saved as `{model_name}_best.pt`
- Class mapping saved as `{model_name}_classes.json`
- All models saved in `ml/models/` directory

---

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `--batch-size` or `--data-fraction`
2. **Slow Training**: Increase `--num-workers` or use `--device cuda`
3. **Poor Accuracy**: Increase `--epochs` or `--data-fraction`
4. **Overfitting**: Reduce `--data-fraction` or increase `--weight-decay`

### Performance Tips
- Use GPU when available (`--device cuda`)
- Start with smaller models for testing (`efficientnet_b0`)
- Use fast training for initial experiments
- Scale up to improved/enhanced for final models
