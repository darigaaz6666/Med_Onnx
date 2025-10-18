import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self, in_features=4, hidden=16, out_features=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 8),
            nn.ReLU(),
            nn.Linear(8, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
