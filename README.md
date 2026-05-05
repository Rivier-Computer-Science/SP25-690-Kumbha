# Learning When NOT to Predict: Selective Classification with Reject Option

## Overview

This project implements selective classification for image recognition using CIFAR-10. Instead of forcing a prediction on every input, the model learns to abstain from uncertain samples. This reduces risk on accepted predictions and improves reliability under distribution shift.

## Problem Statement

Standard deep learning classifiers assign a class to every input regardless of confidence. This leads to high-confidence errors, particularly on out-of-distribution or noisy data. Selective classification addresses this by introducing a rejection mechanism: the model predicts only when it is sufficiently certain, and abstains otherwise.

The core trade-off is between **coverage** (fraction of samples accepted) and **risk** (error rate on accepted samples). A risk-coverage curve characterizes this trade-off and serves as the primary evaluation tool.

## Models

### CNN Baseline
A ResNet-style deep CNN with 4 residual blocks and global average pooling. Uses standard softmax cross-entropy without any rejection capability. Provides the baseline accuracy and risk-coverage curve using maximum softmax probability as confidence.

### CNN with Selective Classification
Same CNN backbone augmented with a dedicated selection head. The selection head produces a scalar score in [0, 1] representing the model's willingness to predict. Trained with a joint selective risk objective that penalizes low coverage.

### Vision Transformer with Selective Classification
A custom Vision Transformer with patch size 4, depth 6, 3 attention heads, and embedding dimension 192. The [CLS] token representation is used for both classification and selection. Trained with the same selective risk objective and a warmup + cosine annealing schedule.

## Directory Structure

```
selective_classification/
    config.py
    main.py
    requirements.txt
    README.md
    data/
        data_loader.py
    models/
        cnn.py
        vit.py
    training/
        trainer.py
    scripts/
        train_cnn_baseline.py
        train_cnn_selective.py
        train_vit_selective.py
        evaluate_thresholds.py
    utils/
        metrics.py
        utils.py
    outputs/
        checkpoints/
        plots/
        results/
```

## Installation

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Running the Project

**Full pipeline (train all models + evaluate):**
```bash
python main.py
```

**Train individual models:**
```bash
python main.py --mode train_cnn_baseline
python main.py --mode train_cnn_selective
python main.py --mode train_vit_selective
```

**Evaluate only (after training):**
```bash
python main.py --mode evaluate
```

**Skip training, run evaluation with existing checkpoints:**
```bash
python main.py --skip_training
```

## Experiments

| Experiment | Description |
|---|---|
| Clean evaluation | All models on CIFAR-10 test set, no noise |
| Noisy evaluation | Gaussian noise added at levels 0.1, 0.2, 0.3, 0.5 |
| Threshold sweep | 100 thresholds from 0 to 1, full risk-coverage curve |
| Model comparison | CNN Baseline vs CNN Selective vs ViT Selective |
| Noise robustness | Accuracy and coverage plotted against noise level |

## Metrics

- **Coverage**: Fraction of test samples accepted (not rejected)
- **Risk**: 1 - accuracy on accepted samples
- **Accuracy**: Correct predictions over accepted samples
- **AURC**: Area under risk-coverage curve (lower is better)
- **E-AURC**: Excess AURC over the optimal baseline
- **Risk at Coverage**: Risk when coverage is fixed to a target level

## Outputs

After running `python main.py`, the following are generated:

**Checkpoints** (`outputs/checkpoints/`):
- `cnn_baseline_best.pt`
- `cnn_selective_best.pt`
- `vit_selective_best.pt`

**Plots** (`outputs/plots/`):
- `cnn_baseline_training_curves.png`
- `cnn_selective_training_curves.png`
- `vit_selective_training_curves.png`
- `risk_coverage_clean.png`
- `risk_coverage_noise_0.1.png`
- `risk_coverage_noise_0.3.png`
- `noise_comparison.png`
- `threshold_sweep_cnn_baseline.png`
- `threshold_sweep_cnn_selective.png`
- `threshold_sweep_vit_selective.png`

**Results** (`outputs/results/`):
- `cnn_baseline_training_history.json`
- `cnn_selective_training_history.json`
- `vit_selective_training_history.json`
- `evaluation_clean.json`
- `noise_evaluation_results.json`

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 40 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 1e-3 (CNN), 3e-4 (ViT) |
| Weight decay | 1e-4 |
| LR scheduler | Cosine annealing (ViT: warmup + cosine) |
| Selective loss lambda | 0.5 |
| Target coverage | 0.8 |

## Loss Function

The selective models are trained with a joint objective:

```
L = E[g(x) * CE(f(x), y)] / E[g(x)] + lambda * max(0, kappa - E[g(x)])^2
```

Where `g(x)` is the selection score, `f(x)` is the classifier output, `CE` is cross-entropy, `kappa` is the target coverage, and `lambda` controls the coverage penalty strength.
