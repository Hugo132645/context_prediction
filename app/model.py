import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchPositionNet(nn.Module):
    def __init__(self):
        super(PatchPositionNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),  #2 patches stacked → 6 channels
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 8)  #we can adjust patch directions here
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
