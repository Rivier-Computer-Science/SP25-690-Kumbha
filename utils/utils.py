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
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['lr'],
                                   weight_decay=config['training']['weight_decay'])

    def train_epoch(self, trainloader):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits, confidence = self.model(inputs)
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
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits, confidence = self.model(inputs)
                preds = logits.argmax(dim=1)
                conf_vals = confidence.squeeze()
                accept_mask = conf_vals >= reject_threshold
                accepted += accept_mask.sum().item()
                correct += (preds[accept_mask] == labels[accept_mask]).sum().item()
                total += labels.size(0)
                logits_all.append(logits)
                labels_all.append(labels)
        accuracy = correct / accepted if accepted > 0 else 0.0
        coverage = accepted / total
        risk = 1 - accuracy if accepted > 0 else 1.0
        all_logits = torch.cat(logits_all)
        all_labels = torch.cat(labels_all)
        ece = compute_ece(all_logits, all_labels)
        return accuracy, coverage, risk, ece
