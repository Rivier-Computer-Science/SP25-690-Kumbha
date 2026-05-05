import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for d in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
NUM_CLASSES = 10
IMAGE_SIZE = 32
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CNN_BASELINE_CONFIG = {
    "model_name": "cnn_baseline",
    "num_classes": NUM_CLASSES,
    "dropout_rate": 0.3,
    "use_rejection": False,
}

CNN_SELECTIVE_CONFIG = {
    "model_name": "cnn_selective",
    "num_classes": NUM_CLASSES,
    "dropout_rate": 0.3,
    "use_rejection": True,
}

VIT_SELECTIVE_CONFIG = {
    "model_name": "vit_selective",
    "num_classes": NUM_CLASSES,
    "image_size": IMAGE_SIZE,
    "patch_size": 4,
    "embed_dim": 192,
    "num_heads": 3,
    "depth": 6,
    "mlp_ratio": 4,
    "dropout_rate": 0.1,
    "use_rejection": True,
}

TRAIN_CONFIG = {
    "batch_size": 128,
    "num_epochs": 40,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler": "cosine",
    "warmup_epochs": 5,
    "num_workers": 2,
    "pin_memory": True,
}

SELECTIVE_TRAIN_CONFIG = {
    **TRAIN_CONFIG,
    "coverage_lambda": 0.5,
    "target_coverage": 0.8,
}

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
THRESHOLD_RANGE = (0.0, 1.0)
THRESHOLD_STEPS = 100
EVAL_COVERAGE_TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]