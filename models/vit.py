import torch
import torch.nn as nn
import timm

class ViTSelective(nn.Module):
    def __init__(self, num_classes=10, img_size=224):
        super(ViTSelective, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, 
                                     num_classes=num_classes, img_size=img_size)
        self.confidence = nn.Linear(self.vit.num_features, 1)

    def forward(self, x):
        features = self.vit.forward_features(x)
        features = features[:, 0]  # CLS token
        logits = self.vit.head(features)
        confidence = torch.sigmoid(self.confidence(features))
        return logits, confidence
