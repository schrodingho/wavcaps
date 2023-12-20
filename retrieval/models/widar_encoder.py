import torch
import einops
import numpy as np
import torch.nn as nn

class Widar_CNN5(nn.Module):
    def __init__(self, hidden_dim=1024, output_dim=512):
        super(Widar_CNN5, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (22,20,20)
            nn.Conv2d(22, 32, 6, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 96, 3, stride=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 96 * 4 * 4)
        x = self.fc(x)
        return x