import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import compute_ece

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=config['training']['lr'],
                                   weight_decay=config['training']['weight_decay'])

    def train_epoch(self, trainloader):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(trainloader)

    def evaluate(self, testloader, reject_threshold=0.0):
        self.model.eval()
        correct = 0
        total = 0
        accepted = 0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(self.device)
                logits, confidence = self.model(inputs)
                preds = logits.argmax(dim=1)
                conf_vals = confidence.squeeze()
                accept_mask = conf_vals >= reject_threshold
                accepted += accept_mask.sum().item()
                correct += (preds[accept_mask] == labels.to(self.device)[accept_mask]).sum().item()
                total += labels.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        accuracy = correct / accepted if accepted > 0 else 0.0
        coverage = accepted / total if total > 0 else 0.0
        risk = 1 - accuracy if accepted > 0 else 1.0
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        ece = compute_ece(all_logits, all_labels)
        return accuracy, coverage, risk, ece
