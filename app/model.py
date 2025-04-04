import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class PatchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            ResidualBlock(256),  # added residual block
            nn.AdaptiveAvgPool2d((3, 3))
        )

    def forward(self, x):
        return self.layers(x)

class SiameseContextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = PatchFeatureExtractor()

        # Main classification head: 8 directions
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 8)
        )

        # Optional auxiliary task: same-object prediction (binary)
        self.aux_head = nn.Sequential(
            nn.Linear(2 * 256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        p1, p2 = x[:, :3], x[:, 3:]
        f1 = self.feature_extractor(p1)
        f2 = self.feature_extractor(p2)
        fused = torch.cat([f1, f2], dim=1)

        fused_flat = fused.view(fused.size(0), -1)
        direction_logits = self.classifier(fused_flat)
        same_object_prob = self.aux_head(fused_flat)

        return direction_logits, same_object_prob
