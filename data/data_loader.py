import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from config import CIFAR10_MEAN, CIFAR10_STD, DATA_DIR, TRAIN_CONFIG


class NoisyDataset(Dataset):
    def __init__(self, base_dataset, noise_level=0.0):
        self.base_dataset = base_dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.noise_level > 0.0:
            noise = torch.randn_like(image) * self.noise_level
            image = torch.clamp(image + noise, -3.0, 3.0)
        return image, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_cifar10_loaders(batch_size=None, num_workers=None, noise_level=0.0):
    if batch_size is None:
        batch_size = TRAIN_CONFIG["batch_size"]
    if num_workers is None:
        num_workers = TRAIN_CONFIG["num_workers"]

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    if noise_level > 0.0:
        test_dataset = NoisyDataset(test_dataset, noise_level=noise_level)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG["pin_memory"],
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG["pin_memory"],
    )

    return train_loader, test_loader


def get_noisy_test_loaders(batch_size=None, num_workers=None, noise_levels=None):
    if noise_levels is None:
        from config import NOISE_LEVELS
        noise_levels = NOISE_LEVELS
    if batch_size is None:
        batch_size = TRAIN_CONFIG["batch_size"]
    if num_workers is None:
        num_workers = TRAIN_CONFIG["num_workers"]

    test_transform = get_transforms(train=False)
    base_test = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    loaders = {}
    for nl in noise_levels:
        if nl > 0.0:
            dataset = NoisyDataset(base_test, noise_level=nl)
        else:
            dataset = base_test
        loaders[nl] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=TRAIN_CONFIG["pin_memory"],
        )
    return loaders