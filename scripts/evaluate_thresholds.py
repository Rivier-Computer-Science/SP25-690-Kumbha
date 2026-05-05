import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from config import (
    CNN_BASELINE_CONFIG, CNN_SELECTIVE_CONFIG, VIT_SELECTIVE_CONFIG,
    SELECTIVE_TRAIN_CONFIG, TRAIN_CONFIG, CHECKPOINT_DIR, NOISE_LEVELS,
    THRESHOLD_STEPS, EVAL_COVERAGE_TARGETS,
)
from data.data_loader import get_cifar10_loaders, get_noisy_test_loaders
from models.cnn import CNNBackbone
from models.vit import VisionTransformer
from training.trainer import BaselineTrainer, SelectiveTrainer
from utils.metrics import (
    compute_risk_coverage_curve,
    compute_selective_metrics,
    compute_aurc,
    compute_eaurc,
    compute_risk_at_coverage,
    extract_confidence_scores,
)
from utils.utils import (
    set_seed, get_device, save_results,
    plot_risk_coverage_curve, plot_noise_comparison, plot_threshold_sweep,
    load_checkpoint,
)


def load_cnn_baseline(device):
    model = CNNBackbone(
        num_classes=CNN_BASELINE_CONFIG["num_classes"],
        dropout_rate=CNN_BASELINE_CONFIG["dropout_rate"],
        use_rejection=False,
    )
    ckpt_path = os.path.join(CHECKPOINT_DIR, "cnn_baseline_best.pt")
    if os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path)
        print("Loaded CNN Baseline checkpoint.")
    else:
        print("WARNING: CNN Baseline checkpoint not found. Using random weights.")
    model.to(device)
    model.eval()
    return model


def load_cnn_selective(device):
    model = CNNBackbone(
        num_classes=CNN_SELECTIVE_CONFIG["num_classes"],
        dropout_rate=CNN_SELECTIVE_CONFIG["dropout_rate"],
        use_rejection=True,
    )
    ckpt_path = os.path.join(CHECKPOINT_DIR, "cnn_selective_best.pt")
    if os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path)
        print("Loaded CNN Selective checkpoint.")
    else:
        print("WARNING: CNN Selective checkpoint not found. Using random weights.")
    model.to(device)
    model.eval()
    return model


def load_vit_selective(device):
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
    ckpt_path = os.path.join(CHECKPOINT_DIR, "vit_selective_best.pt")
    if os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path)
        print("Loaded ViT Selective checkpoint.")
    else:
        print("WARNING: ViT Selective checkpoint not found. Using random weights.")
    model.to(device)
    model.eval()
    return model


def collect_predictions_baseline(model, loader, device):
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting", leave=False):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    confidences, predictions = extract_confidence_scores(all_logits, method="max_softmax")
    return confidences, predictions, all_labels


def collect_predictions_selective(model, loader, device):
    all_logits = []
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting", leave=False):
            images = images.to(device)
            logits, scores = model(images)
            all_logits.append(logits.cpu())
            all_scores.append(scores.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits, dim=0)
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    _, predictions = extract_confidence_scores(all_logits, method="max_softmax")
    return all_scores, predictions, all_labels


def evaluate_all_thresholds():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print("=" * 60)
    print("Evaluating All Models with Threshold Sweep")
    print("=" * 60)

    cnn_baseline = load_cnn_baseline(device)
    cnn_selective = load_cnn_selective(device)
    vit_selective = load_vit_selective(device)

    _, clean_test_loader = get_cifar10_loaders(noise_level=0.0)
    noisy_loaders = get_noisy_test_loaders(noise_levels=NOISE_LEVELS)

    print("\n--- Clean Data Evaluation ---")

    conf_base, pred_base, labels_base = collect_predictions_baseline(cnn_baseline, clean_test_loader, device)
    conf_cnn_sel, pred_cnn_sel, labels_cnn_sel = collect_predictions_selective(cnn_selective, clean_test_loader, device)
    conf_vit_sel, pred_vit_sel, labels_vit_sel = collect_predictions_selective(vit_selective, clean_test_loader, device)

    rc_baseline = compute_risk_coverage_curve(conf_base, pred_base, labels_base, THRESHOLD_STEPS)
    rc_cnn_sel = compute_risk_coverage_curve(conf_cnn_sel, pred_cnn_sel, labels_cnn_sel, THRESHOLD_STEPS)
    rc_vit_sel = compute_risk_coverage_curve(conf_vit_sel, pred_vit_sel, labels_vit_sel, THRESHOLD_STEPS)

    rc_curves_clean = {
        "CNN Baseline": rc_baseline,
        "CNN Selective": rc_cnn_sel,
        "ViT Selective": rc_vit_sel,
    }

    plot_risk_coverage_curve(rc_curves_clean, "Clean Test Data", "risk_coverage_clean.png")
    print("Saved: risk_coverage_clean.png")

    for model_name, rc_data in [("CNN_Baseline", rc_baseline), ("CNN_Selective", rc_cnn_sel), ("ViT_Selective", rc_vit_sel)]:
        ts_path = plot_threshold_sweep(rc_data, model_name, f"threshold_sweep_{model_name.lower()}.png")
        print(f"Saved threshold sweep: threshold_sweep_{model_name.lower()}.png")

    print("\n--- Summary at Coverage Targets (Clean) ---")
    summary_clean = {}
    for model_name, rc_data in [("CNN Baseline", rc_baseline), ("CNN Selective", rc_cnn_sel), ("ViT Selective", rc_vit_sel)]:
        coverages = [r["coverage"] for r in rc_data]
        risks = [r["risk"] for r in rc_data]
        aurc = compute_aurc(coverages, risks)
        eaurc = compute_eaurc(coverages, risks)
        coverage_targets_results = {}
        for cov_target in EVAL_COVERAGE_TARGETS:
            risk_at_cov = compute_risk_at_coverage(coverages, risks, cov_target)
            coverage_targets_results[cov_target] = risk_at_cov
        summary_clean[model_name] = {
            "aurc": aurc,
            "eaurc": eaurc,
            "risk_at_coverage": coverage_targets_results,
        }
        print(f"\n{model_name}:")
        print(f"  AURC: {aurc:.4f}, E-AURC: {eaurc:.4f}")
        for cov, risk in coverage_targets_results.items():
            print(f"  Risk @ {cov*100:.0f}% coverage: {risk*100:.2f}%")

    save_results({"rc_clean": {k: v for k, v in zip(
        ["cnn_baseline", "cnn_selective", "vit_selective"],
        [rc_baseline, rc_cnn_sel, rc_vit_sel]
    )}, "summary_clean": summary_clean}, "evaluation_clean.json")

    print("\n--- Noisy Data Evaluation ---")
    noise_results = {"CNN Baseline": {}, "CNN Selective": {}, "ViT Selective": {}}

    for nl in NOISE_LEVELS:
        loader = noisy_loaders[nl]
        print(f"\n  Noise Level: {nl}")

        c, p, l = collect_predictions_baseline(cnn_baseline, loader, device)
        m = compute_selective_metrics(c, p, l, threshold=0.5)
        noise_results["CNN Baseline"][nl] = {"accuracy": 1.0 - m["risk"], "coverage": m["coverage"]}

        c, p, l = collect_predictions_selective(cnn_selective, loader, device)
        m = compute_selective_metrics(c, p, l, threshold=0.5)
        noise_results["CNN Selective"][nl] = {"accuracy": 1.0 - m["risk"], "coverage": m["coverage"]}

        c, p, l = collect_predictions_selective(vit_selective, loader, device)
        m = compute_selective_metrics(c, p, l, threshold=0.5)
        noise_results["ViT Selective"][nl] = {"accuracy": 1.0 - m["risk"], "coverage": m["coverage"]}

        print(f"    CNN Baseline   Acc: {noise_results['CNN Baseline'][nl]['accuracy']*100:.2f}%  Cov: {noise_results['CNN Baseline'][nl]['coverage']*100:.2f}%")
        print(f"    CNN Selective  Acc: {noise_results['CNN Selective'][nl]['accuracy']*100:.2f}%  Cov: {noise_results['CNN Selective'][nl]['coverage']*100:.2f}%")
        print(f"    ViT Selective  Acc: {noise_results['ViT Selective'][nl]['accuracy']*100:.2f}%  Cov: {noise_results['ViT Selective'][nl]['coverage']*100:.2f}%")

    plot_noise_comparison(noise_results, ["CNN Baseline", "CNN Selective", "ViT Selective"], "noise_comparison.png")
    print("\nSaved: noise_comparison.png")

    print("\n--- Noisy Data Risk-Coverage Curves ---")
    for nl in [0.1, 0.3]:
        loader = noisy_loaders.get(nl)
        if loader is None:
            continue
        c_b, p_b, l_b = collect_predictions_baseline(cnn_baseline, loader, device)
        c_cs, p_cs, l_cs = collect_predictions_selective(cnn_selective, loader, device)
        c_vs, p_vs, l_vs = collect_predictions_selective(vit_selective, loader, device)

        rc_noisy = {
            "CNN Baseline": compute_risk_coverage_curve(c_b, p_b, l_b, THRESHOLD_STEPS),
            "CNN Selective": compute_risk_coverage_curve(c_cs, p_cs, l_cs, THRESHOLD_STEPS),
            "ViT Selective": compute_risk_coverage_curve(c_vs, p_vs, l_vs, THRESHOLD_STEPS),
        }
        plot_risk_coverage_curve(rc_noisy, f"Noise Level={nl}", f"risk_coverage_noise_{nl}.png")
        print(f"Saved: risk_coverage_noise_{nl}.png")

    save_results(noise_results, "noise_evaluation_results.json")
    print("\nAll evaluations complete. Results saved.")


if __name__ == "__main__":
    evaluate_all_thresholds()
