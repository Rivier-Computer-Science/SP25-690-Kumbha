import torch
from torchvision import datasets, transforms
import torchvision
import numpy as np
from PIL import Image
import os

def get_transforms(train=True, img_size=32):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

def generate_corrupted_image(img, corruption_type, severity):
    img = np.array(img).astype(np.float32) / 255.0
    if corruption_type == 'gaussian_noise':
        noise = np.random.normal(0, severity * 0.1, img.shape)
        img = np.clip(img + noise, 0, 1)
    elif corruption_type == 'blur':
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=severity * 0.5)
    elif corruption_type == 'brightness':
        img = np.clip(img * (1 + severity * 0.2), 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

class CorruptedCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True, corruption_prob=0.3, corruptions=None):
        super().__init__(root, train=train, transform=transform, download=download)
        self.corruption_prob = corruption_prob
        self.corruptions = corruptions or ['gaussian_noise', 'blur', 'brightness']

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if np.random.rand() < self.corruption_prob and self.corruptions:
            corr_type = np.random.choice(self.corruptions)
            severity = np.random.choice([1, 3, 5])
            img = generate_corrupted_image(img, corr_type, severity)
            if self.transform:
                img = self.transform(img)
        return img, target

def get_dataloaders(config):
    transform_train = get_transforms(train=True)
    transform_test = get_transforms(train=False)
    
    trainset = CorruptedCIFAR10(root=config['dataset']['root'], train=True, 
                               transform=transform_train, download=True)
    testset = CorruptedCIFAR10(root=config['dataset']['root'], train=False, 
                              transform=transform_test, download=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['dataset']['batch_size'],
                                             shuffle=True, num_workers=config['dataset']['num_workers'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['dataset']['batch_size'],
                                            shuffle=False, num_workers=config['dataset']['num_workers'])
    return trainloader, testloader
