import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import argparse

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ======================
# DATA
# ======================
TRAINING_MODES = {
    "fast": {
        "subset_size": 5000,
        "cnn_epochs": 3,
        "vit_epochs": 2,
    },
    "complete": {
        "subset_size": None,
        "cnn_epochs": 20,
        "vit_epochs": 10,
    },
}

transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_data_loaders(mode="fast"):
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {mode}")

    subset_size = TRAINING_MODES[mode]["subset_size"]

    train_cnn = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_cnn
    )
    test_cnn = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_cnn
    )

    train_vit = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_vit
    )
    test_vit = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_vit
    )

    if subset_size is not None:
        indices = torch.randperm(len(train_cnn))[:subset_size]
        train_cnn = torch.utils.data.Subset(train_cnn, indices)
        train_vit = torch.utils.data.Subset(train_vit, indices)

    train_loader_cnn = torch.utils.data.DataLoader(train_cnn, batch_size=128, shuffle=True)
    test_loader_cnn = torch.utils.data.DataLoader(test_cnn, batch_size=128)
    train_loader_vit = torch.utils.data.DataLoader(train_vit, batch_size=64, shuffle=True)
    test_loader_vit = torch.utils.data.DataLoader(test_vit, batch_size=64)

    return train_loader_cnn, test_loader_cnn, train_loader_vit, test_loader_vit


# ======================
# CNN MODEL
# ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# ViT MODEL
# ======================
def get_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    return model


# ======================
# TRAIN
# ======================
def train(model, loader, epochs=3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epochs):
        model.train()
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {e+1}: Loss {total:.3f}")

    return model


# ======================
# SELECTIVE CLASSIFICATION
# ======================
def selective(logits, threshold):
    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return pred, conf >= threshold
# ======================
# NOISE FUNCTION (NEW)
# ======================
def add_noise(x, noise_level=0.1):
    noise = torch.randn_like(x) * noise_level
    return torch.clamp(x + noise, 0, 1)

# ======================
# EVALUATION
# ======================
def evaluate(model, loader, threshold=0.7):
    model.eval()

    accs, covs, risks = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            x = add_noise(x)   # NEW LINE (adds uncertainty)
            logits = model(x)
            pred, accept = selective(logits, threshold)

            acc_mask = accept.cpu().numpy()
            y_true = y.cpu().numpy()
            y_pred = pred.cpu().numpy()

            if acc_mask.sum() == 0:
                continue

            acc = accuracy_score(y_true[acc_mask], y_pred[acc_mask])
            cov = acc_mask.mean()
            risk = 1 - acc

            accs.append(acc)
            covs.append(cov)
            risks.append(risk)

    return np.mean(accs), np.mean(covs), np.mean(risks)


def inference_metrics(model, loader, threshold=0.7):
    model.eval()

    all_preds = []
    all_labels = []
    all_conf = []
    all_accept = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = add_noise(x)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            accept = conf >= threshold

            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
            all_conf.append(conf.cpu())
            all_accept.append(accept.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_conf = torch.cat(all_conf).numpy()
    all_accept = torch.cat(all_accept).numpy()

    overall_acc = accuracy_score(all_labels, all_preds)
    coverage = all_accept.mean()
    accepted_count = all_accept.sum()
    accepted_acc = accuracy_score(all_labels[all_accept], all_preds[all_accept]) if accepted_count > 0 else 0.0
    rejected_rate = 1.0 - coverage
    risk = 1.0 - accepted_acc
    average_confidence = all_conf.mean()

    return {
        "overall_accuracy": overall_acc,
        "accepted_accuracy": accepted_acc,
        "coverage": coverage,
        "rejected_rate": rejected_rate,
        "risk": risk,
        "average_confidence": average_confidence,
        "accepted_count": int(accepted_count),
        "total_count": int(len(all_labels)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective classification training")
    parser.add_argument(
        "--mode",
        choices=list(TRAINING_MODES.keys()),
        default="fast",
        help="Training mode: fast experiment (default) or complete training over many epochs.",
    )
    args = parser.parse_args()

    print(f"Using training mode: {args.mode}")
    train_loader_cnn, test_loader_cnn, train_loader_vit, test_loader_vit = get_data_loaders(args.mode)
    cnn_epochs = TRAINING_MODES[args.mode]["cnn_epochs"]
    vit_epochs = TRAINING_MODES[args.mode]["vit_epochs"]

    print("Training CNN...")
    cnn = train(CNN(), train_loader_cnn, epochs=cnn_epochs)

    print("Training ViT...")
    vit = train(get_vit(), train_loader_vit, epochs=vit_epochs)

    # ======================
    # RISK-COVERAGE CURVE
    # ======================
    thresholds = np.linspace(0, 1, 10)

    cnn_cov, cnn_risk = [], []
    vit_cov, vit_risk = [], []

    for t in thresholds:
        _, c1, r1 = evaluate(cnn, test_loader_cnn, t)
        cnn_cov.append(c1)
        cnn_risk.append(r1)

        _, c2, r2 = evaluate(vit, test_loader_vit, t)
        vit_cov.append(c2)
        vit_risk.append(r2)


    # ======================
    # PLOT
    # ======================
    plt.plot(cnn_cov, cnn_risk, label="CNN")
    plt.plot(vit_cov, vit_risk, label="ViT")

    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title("Selective Classification: Risk-Coverage")
    plt.legend()
    plt.show()


    # ======================
    # FINAL OUTPUT
    # ======================
    print("\nFinal Results (threshold=0.7)")
    cnn_selective = evaluate(cnn, test_loader_cnn, 0.7)
    vit_selective = evaluate(vit, test_loader_vit, 0.7)
    print("CNN selective (acc, coverage, risk):", cnn_selective)
    print("ViT selective (acc, coverage, risk):", vit_selective)

    print("\nInference comparison at threshold=0.7")
    cnn_inference = inference_metrics(cnn, test_loader_cnn, 0.7)
    vit_inference = inference_metrics(vit, test_loader_vit, 0.7)

    print("CNN inference metrics:")
    print(f"  Overall accuracy:    {cnn_inference['overall_accuracy']:.4f}")
    print(f"  Accepted accuracy:   {cnn_inference['accepted_accuracy']:.4f}")
    print(f"  Coverage:            {cnn_inference['coverage']:.4f}")
    print(f"  Rejection rate:      {cnn_inference['rejected_rate']:.4f}")
    print(f"  Risk:                {cnn_inference['risk']:.4f}")
    print(f"  Avg confidence:      {cnn_inference['average_confidence']:.4f}")
    print(f"  Accepted / total:    {cnn_inference['accepted_count']} / {cnn_inference['total_count']}")

    print("\nViT inference metrics:")
    print(f"  Overall accuracy:    {vit_inference['overall_accuracy']:.4f}")
    print(f"  Accepted accuracy:   {vit_inference['accepted_accuracy']:.4f}")
    print(f"  Coverage:            {vit_inference['coverage']:.4f}")
    print(f"  Rejection rate:      {vit_inference['rejected_rate']:.4f}")
    print(f"  Risk:                {vit_inference['risk']:.4f}")
    print(f"  Avg confidence:      {vit_inference['average_confidence']:.4f}")
    print(f"  Accepted / total:    {vit_inference['accepted_count']} / {vit_inference['total_count']}")
