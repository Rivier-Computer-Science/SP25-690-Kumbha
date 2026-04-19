import torch
import numpy as np
from sklearn.calibration import calibration_curve

def compute_risk_coverage(logits_list, confidence_list, labels_list, thresholds):
    risks = []
    coverages = []
    for thresh in thresholds:
        accepted = np.array(confidence_list) >= thresh
        if np.sum(accepted) == 0:
            risks.append(1.0)
            coverages.append(0.0)
            continue
        preds = np.argmax(np.array(logits_list), axis=1)
        correct = preds[accepted] == np.array(labels_list)[accepted]
        risk = 1 - np.mean(correct)
        coverage = np.mean(accepted)
        risks.append(risk)
        coverages.append(coverage)
    return np.array(risks), np.array(coverages)

def compute_ece(logits, labels, n_bins=15):
    confidences = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
    predictions = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if np.any(in_bin):
            acc_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * np.mean(in_bin)
    return ece
