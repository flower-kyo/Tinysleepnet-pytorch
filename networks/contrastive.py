import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    projection head for contrastive learning.
    """
    def __init__(self, dim_mlp, dim_nce, l2_norm=True):
        super(MLP, self).__init__()
        self.l2_norm = l2_norm
        self.mlp = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.linear(dim_mlp, dim_nce))

    def forward(self, x):
        if self.l2_norm:
            x = nn.functional.normalize(x, dim=1)
        x = self.mlp(x)
        return x