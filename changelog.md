# 04/19/2026 00:00 Initial Setup

- Added main.py: PyTorch implementation for selective classification using ResNet18 (CNN) and Vision Transformer (ViT) on CIFAR-10 dataset
- Implemented data loading with image transforms (resize, flip, to tensor) and subset sampling for faster training (5000 samples)
- Added training function with Adam optimizer, freezing pretrained layers except final classification heads
- Implemented evaluation with confidence-based rejection mechanism using adjustable thresholds (0.5, 0.7, 0.9)
- Metrics calculation: coverage, accuracy on accepted samples, and risk (error rate on accepted samples)
- Added README.md with comprehensive project description including problem statement, method, baseline, experimental setup, and expected results
- Included LICENSE file
- Pre-downloaded CIFAR-10 dataset in data/cifar-10-batches-py/ with training batches, test batch, and metadata