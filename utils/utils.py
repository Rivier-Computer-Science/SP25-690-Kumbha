import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import PLOT_DIR, RESULTS_DIR, CHECKPOINT_DIR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, metrics, model_name, scheduler=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_epoch{epoch}.pt")
    torch.save(checkpoint, path)
    best_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    torch.save(checkpoint, best_path)
    return best_path


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def save_results(results_dict, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)

    def convert(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [deep_convert(v) for v in d]
        else:
            return convert(d)

    with open(path, "w") as f:
        json.dump(deep_convert(results_dict), f, indent=2)
    return path


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#2563eb", linewidth=2)
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in train_accs], label="Train Acc", color="#2563eb", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in val_accs], label="Val Acc", color="#dc2626", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_risk_coverage_curve(rc_results_dict, title, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#db2777"]

    for idx, (model_name, rc_data) in enumerate(rc_results_dict.items()):
        color = colors[idx % len(colors)]
        coverages = [r["coverage"] for r in rc_data]
        risks = [r["risk"] for r in rc_data]
        sorted_pairs = sorted(zip(coverages, risks))
        coverages = [p[0] for p in sorted_pairs]
        risks = [p[1] for p in sorted_pairs]

        axes[0].plot(coverages, risks, label=model_name, color=color, linewidth=2)
        axes[1].plot(coverages, [1.0 - r for r in risks], label=model_name, color=color, linewidth=2)

    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Risk (1 - Accuracy)")
    axes[0].set_title(f"{title} - Risk vs Coverage")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    axes[1].set_xlabel("Coverage")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Accuracy vs Coverage")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_noise_comparison(noise_results, model_names, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    markers = ["o", "s", "^"]

    for idx, model_name in enumerate(model_names):
        if model_name not in noise_results:
            continue
        noise_data = noise_results[model_name]
        noise_levels = sorted(noise_data.keys())
        accs = [noise_data[nl]["accuracy"] * 100 for nl in noise_levels]
        coverages = [noise_data[nl].get("coverage", 1.0) * 100 for nl in noise_levels]

        axes[0].plot(noise_levels, accs, label=model_name, color=colors[idx % len(colors)],
                     marker=markers[idx % len(markers)], linewidth=2, markersize=7)
        axes[1].plot(noise_levels, coverages, label=model_name, color=colors[idx % len(colors)],
                     marker=markers[idx % len(markers)], linewidth=2, markersize=7)

    axes[0].set_xlabel("Noise Level")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy vs Noise Level")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Noise Level")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("Coverage vs Noise Level")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_threshold_sweep(threshold_results, model_name, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    thresholds = [r["threshold"] for r in threshold_results]
    coverages = [r["coverage"] for r in threshold_results]
    risks = [r["risk"] for r in threshold_results]
    accs = [r["accuracy"] * 100 for r in threshold_results]

    axes[0].plot(thresholds, coverages, color="#2563eb", linewidth=2)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Coverage")
    axes[0].set_title(f"{model_name} - Coverage vs Threshold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thresholds, risks, color="#dc2626", linewidth=2)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Risk")
    axes[1].set_title(f"{model_name} - Risk vs Threshold")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(thresholds, accs, color="#16a34a", linewidth=2)
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title(f"{model_name} - Accuracy vs Threshold")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Threshold Sweep: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, model_name):
    total = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"  Trainable parameters: {total:,}")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {params:,}")
    print()
