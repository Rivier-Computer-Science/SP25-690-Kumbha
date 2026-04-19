import torch
import torch.nn as nn
import timm

class ViTSelective(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224', num_classes=10, pretrained=True):
        super(ViTSelective, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, img_size=224)
        self.confidence = nn.Linear(self.vit.num_features, 1)

    def forward(self, x):
        features = self.vit.forward_features(x)
        features = features[:, 0]  # CLS token
        logits = self.vit.head(features)
        confidence = torch.sigmoid(self.confidence(features))
        return logits, confidence
