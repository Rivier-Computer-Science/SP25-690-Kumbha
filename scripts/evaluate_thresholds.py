import torch
import yaml
import numpy as np
import argparse
from models.cnn import CNN
from models.vit import ViTSelective
from utils.data_loader import get_dataloaders
from utils.utils import set_seed, load_model
from utils.metrics import compute_risk_coverage
from training.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vit_selective')
args = parser.parse_args()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

set_seed(config['training']['seed'])
_, testloader = get_dataloaders(config)

if args.model == 'cnn_selective':
    model = CNN(num_classes=config['model']['cnn']['num_classes'])
    load_model(model, "results/checkpoints/cnn_selective.pth", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
elif args.model == 'vit_selective':
    model = ViTSelective(num_classes=config['model']['vit']['num_classes'])
    load_model(model, "results/checkpoints/vit_selective.pth", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

trainer = Trainer(model, config)

thresholds = np.linspace(0.5, 0.99, config['selective']['num_thresholds'])
logits_list = []
conf_list = []
labels_list = []

model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(trainer.device)
        logits, confidence = model(inputs)
        logits_list.extend(logits.cpu().numpy())
        conf_list.extend(confidence.squeeze().cpu().numpy())
        labels_list.extend(labels.numpy())

risks, coverages = compute_risk_coverage(logits_list, conf_list, labels_list, thresholds)
for t, r, c in zip(thresholds, risks, coverages):
    print(f"Threshold: {t:.3f}, Coverage: {c:.4f}, Risk: {r:.4f}")
