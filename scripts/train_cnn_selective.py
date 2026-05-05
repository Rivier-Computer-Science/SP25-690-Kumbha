import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from config import CNN_SELECTIVE_CONFIG, SELECTIVE_TRAIN_CONFIG, CHECKPOINT_DIR
from data.data_loader import get_cifar10_loaders
from models.cnn import CNNBackbone
from training.trainer import SelectiveTrainer
from utils.utils import set_seed, get_device, print_model_summary, plot_training_curves, save_results


def train_cnn_selective():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print("Training CNN with Selective Classification...")

    train_loader, val_loader = get_cifar10_loaders(
        batch_size=SELECTIVE_TRAIN_CONFIG["batch_size"],
        num_workers=SELECTIVE_TRAIN_CONFIG["num_workers"],
    )

    model = CNNBackbone(
        num_classes=CNN_SELECTIVE_CONFIG["num_classes"],
        dropout_rate=CNN_SELECTIVE_CONFIG["dropout_rate"],
        use_rejection=True,
    )

    print_model_summary(model, "CNN Selective")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=SELECTIVE_TRAIN_CONFIG["learning_rate"],
        weight_decay=SELECTIVE_TRAIN_CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SELECTIVE_TRAIN_CONFIG["num_epochs"],
        eta_min=1e-6,
    )

    trainer = SelectiveTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=SELECTIVE_TRAIN_CONFIG,
        model_name="cnn_selective",
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=SELECTIVE_TRAIN_CONFIG["num_epochs"],
    )

    plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        history["train_accs"],
        history["val_accs"],
        model_name="cnn_selective",
    )

    save_results(history, "cnn_selective_training_history.json")

    print(f"\nCNN Selective training complete. Best val acc: {history['best_val_acc']*100:.2f}%")
    return history


if __name__ == "__main__":
    train_cnn_selective()
