import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ======================
# DATA
# ======================
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_cnn = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform_cnn)
test_cnn  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_cnn)

train_vit = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform_vit)
test_vit  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_vit)

train_loader_cnn = torch.utils.data.DataLoader(train_cnn, batch_size=128, shuffle=True)
test_loader_cnn  = torch.utils.data.DataLoader(test_cnn, batch_size=128)

train_loader_vit = torch.utils.data.DataLoader(train_vit, batch_size=64, shuffle=True)
test_loader_vit  = torch.utils.data.DataLoader(test_vit, batch_size=64)


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
# EVALUATION
# ======================
def evaluate(model, loader, threshold=0.7):
    model.eval()

    accs, covs, risks = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

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


# ======================
# TRAIN MODELS
# ======================
print("Training CNN...")
cnn = train(CNN(), train_loader_cnn, epochs=5)

print("Training ViT...")
vit = train(get_vit(), train_loader_vit, epochs=3)


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
print("CNN:", evaluate(cnn, test_loader_cnn, 0.7))
print("ViT:", evaluate(vit, test_loader_vit, 0.7))
