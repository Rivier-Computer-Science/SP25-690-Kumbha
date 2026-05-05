import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.metrics import (
    extract_confidence_scores,
    compute_selective_metrics,
    compute_risk_coverage_curve,
    aggregate_epoch_metrics,
)
from utils.utils import save_checkpoint, get_device


class SelectiveLoss(nn.Module):
    def __init__(self, coverage_lambda=0.5, target_coverage=0.8):
        super().__init__()
        self.coverage_lambda = coverage_lambda
        self.target_coverage = target_coverage
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, selection_scores, labels):
        ce = self.ce_loss(logits, labels)
        weighted_ce = (selection_scores * ce).mean()
        coverage = selection_scores.mean()
        coverage_penalty = torch.clamp(self.target_coverage - coverage, min=0.0) ** 2
        emprical_risk = weighted_ce / (coverage + 1e-8)
        total_loss = emprical_risk + self.coverage_lambda * coverage_penalty
        return total_loss, emprical_risk, coverage_penalty


class BaselineTrainer:
    def __init__(self, model, optimizer, scheduler, config, model_name):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model_name = model_name
        self.device = get_device()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, train_loader):
        self.model.train()
        losses = []
        corrects = []
        totals = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            losses.append(loss.item())
            corrects.append(correct)
            totals.append(labels.size(0))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return aggregate_epoch_metrics(losses, corrects, totals)

    def validate(self, val_loader):
        self.model.eval()
        losses = []
        corrects = []
        totals = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).sum().item()
                losses.append(loss.item())
                corrects.append(correct)
                totals.append(labels.size(0))

        return aggregate_epoch_metrics(losses, corrects, totals)

    def fit(self, train_loader, val_loader, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(
                f"Epoch [{epoch:03d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                metrics = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
                save_checkpoint(self.model, self.optimizer, epoch, metrics, self.model_name, self.scheduler)
                print(f"  Checkpoint saved (best val acc: {self.best_val_acc*100:.2f}%)")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "best_val_acc": self.best_val_acc,
        }

    def evaluate(self, test_loader, num_thresholds=100):
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()

        confidences, predictions = extract_confidence_scores(all_logits, method="max_softmax")
        rc_curve = compute_risk_coverage_curve(confidences, predictions, all_labels, num_thresholds)

        return {
            "rc_curve": rc_curve,
            "confidences": confidences,
            "predictions": predictions,
            "labels": all_labels,
        }


class SelectiveTrainer:
    def __init__(self, model, optimizer, scheduler, config, model_name):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model_name = model_name
        self.device = get_device()
        self.model.to(self.device)
        self.selective_criterion = SelectiveLoss(
            coverage_lambda=config.get("coverage_lambda", 0.5),
            target_coverage=config.get("target_coverage", 0.8),
        )
        self.ce_criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, train_loader):
        self.model.train()
        losses = []
        corrects = []
        totals = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits, selection_scores = self.model(images)
            loss, risk_term, cov_penalty = self.selective_criterion(logits, selection_scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            accepted = selection_scores >= 0.5
            if accepted.sum() > 0:
                correct = (preds[accepted] == labels[accepted]).sum().item()
                n = accepted.sum().item()
            else:
                correct = 0
                n = labels.size(0)
            losses.append(loss.item())
            corrects.append(correct)
            totals.append(n)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "cov": f"{accepted.float().mean().item():.2f}"})

        return aggregate_epoch_metrics(losses, corrects, totals)

    def validate(self, val_loader):
        self.model.eval()
        losses = []
        corrects = []
        totals = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                logits, selection_scores = self.model(images)
                loss, _, _ = self.selective_criterion(logits, selection_scores, labels)
                preds = logits.argmax(dim=-1)
                accepted = selection_scores >= 0.5
                if accepted.sum() > 0:
                    correct = (preds[accepted] == labels[accepted]).sum().item()
                    n = accepted.sum().item()
                else:
                    correct = (preds == labels).sum().item()
                    n = labels.size(0)
                losses.append(loss.item())
                corrects.append(correct)
                totals.append(n)

        return aggregate_epoch_metrics(losses, corrects, totals)

    def fit(self, train_loader, val_loader, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(
                f"Epoch [{epoch:03d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                metrics = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
                save_checkpoint(self.model, self.optimizer, epoch, metrics, self.model_name, self.scheduler)
                print(f"  Checkpoint saved (best val acc: {self.best_val_acc*100:.2f}%)")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "best_val_acc": self.best_val_acc,
        }

    def evaluate(self, test_loader, num_thresholds=100):
        self.model.eval()
        all_logits = []
        all_selection_scores = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                logits, selection_scores = self.model(images)
                all_logits.append(logits.cpu())
                all_selection_scores.append(selection_scores.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_selection_scores = torch.cat(all_selection_scores, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        _, predictions = extract_confidence_scores(all_logits, method="max_softmax")

        rc_curve = compute_risk_coverage_curve(
            all_selection_scores, predictions, all_labels, num_thresholds
        )

        softmax_confidences, _ = extract_confidence_scores(all_logits, method="max_softmax")
        rc_curve_softmax = compute_risk_coverage_curve(
            softmax_confidences, predictions, all_labels, num_thresholds
        )

        return {
            "rc_curve": rc_curve,
            "rc_curve_softmax": rc_curve_softmax,
            "selection_scores": all_selection_scores,
            "softmax_confidences": softmax_confidences,
            "predictions": predictions,
            "labels": all_labels,
        }
