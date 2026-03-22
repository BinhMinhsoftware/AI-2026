"""
PyTorch Model architectures for Dog Breed Classification
"""
import torch
import torch.nn as nn
from torchvision import models

import config_pytorch as config


class DogBreedModel(nn.Module):
    """Dog breed classification model with various backbones"""

    def __init__(self, num_classes=120, architecture='resnet50', pretrained=True):
        """
        Args:
            num_classes: Number of dog breeds
            architecture: Backbone architecture name
            pretrained: Use ImageNet pretrained weights
        """
        super(DogBreedModel, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        # Build model based on architecture
        if architecture == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original FC layer
            self.classifier = self._build_classifier(in_features, num_classes)

        elif architecture == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = self._build_classifier(in_features, num_classes)

        elif architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = self._build_classifier(in_features, num_classes)

        elif architecture == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = self._build_classifier(in_features, num_classes)

        elif architecture == 'vit_b_16':
            self.backbone = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
            self.classifier = self._build_classifier(in_features, num_classes)

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_classifier(self, in_features, num_classes):
        """Build classification head"""
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(config.DROPOUT_RATE / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def freeze_backbone(self):
        """Freeze backbone layers for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone frozen")

    def unfreeze_backbone(self, unfreeze_from=50):
        """Unfreeze backbone layers for fine-tuning"""
        # Get all backbone parameters
        backbone_params = list(self.backbone.parameters())
        total_params = len(backbone_params)

        # Unfreeze last N% of parameters
        unfreeze_count = max(1, int(total_params * unfreeze_from / 100))

        for param in backbone_params[-unfreeze_count:]:
            param.requires_grad = True

        print(f"Unfrozen last {unfreeze_count}/{total_params} backbone parameters")


def create_model(architecture='resnet50', num_classes=120, pretrained=True, freeze_backbone=True):
    """
    Factory function to create model

    Args:
        architecture: Model architecture
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone for transfer learning

    Returns:
        model: PyTorch model
    """
    model = DogBreedModel(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=pretrained
    )

    if freeze_backbone:
        model.freeze_backbone()

    return model


def get_optimizer(model, optimizer_name='adam', lr=0.001):
    """
    Create optimizer

    Args:
        model: PyTorch model
        optimizer_name: Optimizer type
        lr: Learning rate

    Returns:
        optimizer: PyTorch optimizer
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def get_scheduler(optimizer, scheduler_name='reduce_on_plateau'):
    """
    Create learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Scheduler type

    Returns:
        scheduler: PyTorch scheduler
    """
    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=True
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS,
            eta_min=config.MIN_LR
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    return total_params, trainable_params


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...\n")

    architectures = ['resnet50', 'resnet18', 'efficientnet_b0', 'mobilenet_v3']

    for arch in architectures:
        print(f"\n{arch.upper()}:")
        print("-" * 50)
        model = create_model(architecture=arch, num_classes=120, pretrained=True, freeze_backbone=True)
        count_parameters(model)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"Output shape: {output.shape}")
