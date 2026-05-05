import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from config import VIT_SELECTIVE_CONFIG, SELECTIVE_TRAIN_CONFIG
from data.data_loader import get_cifar10_loaders
from models.vit import VisionTransformer
from training.trainer import SelectiveTrainer
from utils.utils import set_seed, get_device, print_model_summary, plot_training_curves, save_results


def train_vit_selective():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print("Training Vision Transformer with Selective Classification...")

    vit_train_config = {
        **SELECTIVE_TRAIN_CONFIG,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "num_epochs": 40,
        "weight_decay": 1e-4,
    }

    train_loader, val_loader = get_cifar10_loaders(
        batch_size=vit_train_config["batch_size"],
        num_workers=vit_train_config["num_workers"],
    )

    model = VisionTransformer(
        image_size=VIT_SELECTIVE_CONFIG["image_size"],
        patch_size=VIT_SELECTIVE_CONFIG["patch_size"],
        num_classes=VIT_SELECTIVE_CONFIG["num_classes"],
        embed_dim=VIT_SELECTIVE_CONFIG["embed_dim"],
        depth=VIT_SELECTIVE_CONFIG["depth"],
        num_heads=VIT_SELECTIVE_CONFIG["num_heads"],
        mlp_ratio=VIT_SELECTIVE_CONFIG["mlp_ratio"],
        dropout_rate=VIT_SELECTIVE_CONFIG["dropout_rate"],
        use_rejection=True,
    )

    print_model_summary(model, "ViT Selective")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=vit_train_config["learning_rate"],
        weight_decay=vit_train_config["weight_decay"],
        betas=(0.9, 0.999),
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=vit_train_config.get("warmup_epochs", 5),
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=vit_train_config["num_epochs"] - vit_train_config.get("warmup_epochs", 5),
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[vit_train_config.get("warmup_epochs", 5)],
    )

    trainer = SelectiveTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=vit_train_config,
        model_name="vit_selective",
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=vit_train_config["num_epochs"],
    )

    plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        history["train_accs"],
        history["val_accs"],
        model_name="vit_selective",
    )

    save_results(history, "vit_selective_training_history.json")

    print(f"\nViT Selective training complete. Best val acc: {history['best_val_acc']*100:.2f}%")
    return history


if __name__ == "__main__":
    train_vit_selective()
