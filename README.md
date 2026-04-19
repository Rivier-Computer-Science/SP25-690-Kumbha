# Learning When NOT to Predict: Selective Classification with Reject Option using CNN and Vision Transformers for Reliable Image Recognition

## Problem Statement and Motivation
In many deep learning systems, models are forced to make predictions even when uncertain, leading to highly confident but incorrect outputs. This is problematic in real-world applications where errors have serious consequences. This project implements selective classification, allowing the model to reject uncertain inputs instead of making unreliable predictions. The goal is to improve reliability by enabling the model to recognize its limitations.

## Task Definition
Input: Image from CIFAR-10 (clean or corrupted with Gaussian noise, blur, or brightness variations).  
Output: Predicted class label or reject decision based on confidence score.  
The system maximizes accuracy on accepted samples while minimizing risk on rejected ones.

## Models
- CNN baseline (forced prediction on all inputs)
- CNN with selective reject option
- Vision Transformer with selective reject option

## Dataset
CIFAR-10 with generated corrupted versions (Gaussian noise, blur, brightness) to simulate real-world uncertainty.

## Metrics
- Accuracy on accepted samples
- Coverage
- Risk (error rate on accepted predictions)
- Expected Calibration Error (ECE)
- Risk-coverage curves

## Repository Structure
- models/: CNN and ViT implementations
- utils/: Data loading, metrics, helpers
- training/: Training and evaluation logic
- scripts/: Training and evaluation scripts
- config.yaml: Hyperparameters
- results/: Checkpoints and plots (generated)

## Installation
pip install -r requirements.txt

## How to Run

1. Train CNN baseline:
python scripts/train_cnn_baseline.py

2. Train CNN with selective classification:
python scripts/train_cnn_selective.py

3. Train Vision Transformer with selective classification:
python scripts/train_vit_selective.py

4. Evaluate with different rejection thresholds:
python scripts/evaluate_thresholds.py --model vit_selective

## Expected Results
Selective models achieve lower risk at the same coverage compared to the baseline. Vision Transformer shows better uncertainty estimation due to global attention.

## Ethics and Responsible Use
The system improves reliability but does not eliminate all errors. Rejected predictions require human review in critical applications. The model should not be used as a fully autonomous decision-making tool without oversight.

## Acknowledgments
Project approved by Professor John Glossner.
