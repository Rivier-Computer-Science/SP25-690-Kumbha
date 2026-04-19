import torch
import yaml
import numpy as np
import argparse
from models.cnn import CNN
from models.vit import ViTSelective
from utils.data_loader import get_dataloaders
from utils.utils import set_seed, load_model
from utils.metrics import compute_risk_coverage, plot_risk_coverage
from training.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vit_selective', choices=['cnn_selective', 'vit_selective'])
args = parser.parse_args()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

set_seed(config['training']['seed'])
_, testloader = get_dataloaders(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == 'cnn_selective':
    model = CNN(num_classes=config['model']['cnn']['num_classes'])
    load_model(model, "results/checkpoints/cnn_selective.pth", device)
    model_name = "CNN_Select"
else:
    model = ViTSelective(num_classes=config['model']['vit']['num_classes'], 
                         img_size=config['model']['vit']['img_size'])
    load_model(model, "results/checkpoints/vit_selective.pth", device)
    model_name = "ViT_Select"

model.to(device)
trainer = Trainer(model, config)

thresholds = np.linspace(0.5, 0.99, config['selective']['num_thresholds'])
logits_list = []
conf_list = []
labels_list = []

model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        logits, confidence = model(inputs)
        logits_list.extend(logits.cpu().numpy())
        conf_list.extend(confidence.squeeze().cpu().numpy())
        labels_list.extend(labels.numpy())

risks, coverages = compute_risk_coverage(logits_list, conf_list, labels_list, thresholds)
plot_risk_coverage(risks, coverages, model_name)

for t, r, c in zip(thresholds, risks, coverages):
    print(f"Threshold: {t:.3f} | Coverage: {c:.4f} | Risk: {r:.4f}")
print(f"Risk-Coverage plot saved to results/plots/risk_coverage_{model_name}.png")
