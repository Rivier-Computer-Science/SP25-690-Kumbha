import numpy as np
import torch


def compute_selective_metrics(confidences, predictions, labels, threshold):
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    labels = np.array(labels)

    accepted_mask = confidences >= threshold
    n_total = len(labels)
    n_accepted = accepted_mask.sum()

    coverage = n_accepted / n_total

    if n_accepted == 0:
        accuracy = 0.0
        risk = 1.0
    else:
        correct = (predictions[accepted_mask] == labels[accepted_mask]).sum()
        accuracy = correct / n_accepted
        risk = 1.0 - accuracy

    return {
        "threshold": threshold,
        "coverage": float(coverage),
        "accuracy": float(accuracy),
        "risk": float(risk),
        "n_accepted": int(n_accepted),
        "n_total": int(n_total),
    }


def compute_risk_coverage_curve(confidences, predictions, labels, num_thresholds=100):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    results = []
    for t in thresholds:
        metrics = compute_selective_metrics(confidences, predictions, labels, t)
        results.append(metrics)
    return results


def compute_aurc(coverages, risks):
    coverages = np.array(coverages)
    risks = np.array(risks)
    sorted_idx = np.argsort(coverages)
    coverages = coverages[sorted_idx]
    risks = risks[sorted_idx]
    aurc = np.trapz(risks, coverages)
    return float(aurc)


def compute_eaurc(coverages, risks):
    coverages = np.array(coverages)
    risks = np.array(risks)
    optimal_risk = risks[coverages == 1.0]
    if len(optimal_risk) == 0:
        optimal_risk = risks[-1]
    else:
        optimal_risk = optimal_risk[0]
    aurc = compute_aurc(coverages, risks)
    eaurc = aurc - optimal_risk
    return float(eaurc)


def compute_coverage_at_risk(coverages, risks, target_risk):
    coverages = np.array(coverages)
    risks = np.array(risks)
    valid = risks <= target_risk
    if not valid.any():
        return 0.0
    return float(coverages[valid].max())


def compute_risk_at_coverage(coverages, risks, target_coverage):
    coverages = np.array(coverages)
    risks = np.array(risks)
    sorted_idx = np.argsort(coverages)
    coverages = coverages[sorted_idx]
    risks = risks[sorted_idx]
    valid = coverages >= target_coverage
    if not valid.any():
        return float(risks[-1])
    return float(risks[valid][0])


def compute_threshold_for_coverage(confidences, target_coverage):
    confidences = np.array(confidences)
    threshold_idx = int((1.0 - target_coverage) * len(confidences))
    sorted_conf = np.sort(confidences)
    threshold_idx = min(threshold_idx, len(sorted_conf) - 1)
    return float(sorted_conf[threshold_idx])


def extract_confidence_scores(logits, method="max_softmax"):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    probs = torch.softmax(logits, dim=-1)
    if method == "max_softmax":
        confidence, predicted = probs.max(dim=-1)
        return confidence.numpy(), predicted.numpy()
    elif method == "entropy":
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        max_entropy = np.log(probs.shape[-1])
        confidence = 1.0 - (entropy / max_entropy)
        predicted = probs.argmax(dim=-1)
        return confidence.numpy(), predicted.numpy()
    else:
        raise ValueError(f"Unknown confidence method: {method}")


def aggregate_epoch_metrics(all_losses, all_corrects, all_totals):
    avg_loss = np.mean(all_losses)
    accuracy = sum(all_corrects) / max(sum(all_totals), 1)
    return avg_loss, accuracy
