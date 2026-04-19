import torch
import yaml
from models.vit import ViTSelective
from utils.data_loader import get_dataloaders
from utils.utils import set_seed, get_device, save_model
from training.trainer import Trainer

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

set_seed(config['training']['seed'])
trainloader, testloader = get_dataloaders(config)
model = ViTSelective(num_classes=config['model']['vit']['num_classes'], pretrained=config['model']['vit']['pretrained'])
trainer = Trainer(model, config)

for epoch in range(config['training']['epochs']):
    loss = trainer.train_epoch(trainloader)
    acc, cov, risk, ece = trainer.evaluate(testloader, reject_threshold=config['selective']['reject_threshold'])
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}, Coverage={cov:.4f}, Risk={risk:.4f}, ECE={ece:.4f}")

save_model(model, "results/checkpoints/vit_selective.pth")
