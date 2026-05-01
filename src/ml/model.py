import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
)


def _make_embedding_head(in_features, embedding_dim, dropout=0.0):
    hidden = max(in_features // 2, embedding_dim * 2)
    layers = [
        nn.Linear(in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden, embedding_dim))
    return nn.Sequential(*layers)


def freeze_early_layers(backbone, prefixes=("0", "1", "2", "3", "4")):
    """
    Freeze layers in an nn.Sequential backbone by index prefix.
    For ResNet wrapped as Sequential(*children()[:-1]):
      index 0 = conv1, 1 = bn1, 2 = relu, 3 = maxpool, 4 = layer1, 5 = layer2
    """
    frozen = []
    for name, param in backbone.named_parameters():
        top_level = name.split(".")[0]
        if top_level in prefixes:
            param.requires_grad = False
            frozen.append(top_level)
    unique = sorted(set(frozen))
    if unique:
        print(f"  Frozen backbone indices: {unique}")
    return unique


class SiameseResnet50(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True, dropout=0.0):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = _make_embedding_head(2048, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseResnet18(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = _make_embedding_head(512, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseResnet34(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = _make_embedding_head(512, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseEfficientNetB0(nn.Module):
    """EfficientNet-B0: ~5.3M params, feature dim 1280. Lightest option."""
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        self.backbone = net.features
        self.pool = net.avgpool
        self.embedding = _make_embedding_head(1280, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseEfficientNetB2(nn.Module):
    """EfficientNet-B2: ~9.1M params, feature dim 1408. Good accuracy/size trade-off."""
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b2(weights=weights)
        self.backbone = net.features
        self.pool = net.avgpool
        self.embedding = _make_embedding_head(1408, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseEfficientNetB3(nn.Module):
    """EfficientNet-B3: ~12M params, feature dim 1536. Strongest of the small EfficientNets."""
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b3(weights=weights)
        self.backbone = net.features
        self.pool = net.avgpool
        self.embedding = _make_embedding_head(1536, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseEfficientNetB4(nn.Module):
    """EfficientNet-B4: ~19M params, feature dim 1792."""
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b4(weights=weights)
        self.backbone = net.features
        self.pool = net.avgpool
        self.embedding = _make_embedding_head(1792, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseEfficientNetB5(nn.Module):
    """EfficientNet-B5: ~30M params, feature dim 2048."""
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b5(weights=weights)
        self.backbone = net.features
        self.pool = net.avgpool
        self.embedding = _make_embedding_head(2048, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseEfficientNetB6(nn.Module):
    """EfficientNet-B6: ~43M params, feature dim 2304."""
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b6(weights=weights)
        self.backbone = net.features
        self.pool = net.avgpool
        self.embedding = _make_embedding_head(2304, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class SiameseEfficientNetB7(nn.Module):
    """EfficientNet-B7: ~66M params, feature dim 2560."""
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b7(weights=weights)
        self.backbone = net.features
        self.pool = net.avgpool
        self.embedding = _make_embedding_head(2560, embedding_dim, dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
