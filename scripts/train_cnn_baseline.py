import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from config import CNN_BASELINE_CONFIG, TRAIN_CONFIG, CHECKPOINT_DIR
from data.data_loader import get_cifar10_loaders
from models.cnn import CNNBackbone
from training.trainer import BaselineTrainer
from utils.utils import set_seed, get_device, print_model_summary, plot_training_curves, save_results


def train_cnn_baseline():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print("Training CNN Baseline (no rejection)...")

    train_loader, val_loader = get_cifar10_loaders(
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=TRAIN_CONFIG["num_workers"],
    )

    model = CNNBackbone(
        num_classes=CNN_BASELINE_CONFIG["num_classes"],
        dropout_rate=CNN_BASELINE_CONFIG["dropout_rate"],
        use_rejection=False,
    )

    print_model_summary(model, "CNN Baseline")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_CONFIG["num_epochs"],
        eta_min=1e-6,
    )

    trainer = BaselineTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=TRAIN_CONFIG,
        model_name="cnn_baseline",
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=TRAIN_CONFIG["num_epochs"],
    )

    plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        history["train_accs"],
        history["val_accs"],
        model_name="cnn_baseline",
    )

    save_results(history, "cnn_baseline_training_history.json")

    print(f"\nCNN Baseline training complete. Best val acc: {history['best_val_acc']*100:.2f}%")
    return history


if __name__ == "__main__":
    train_cnn_baseline()
