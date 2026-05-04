"""Stub: full-version model (not used in paper experiments)."""
import torch.nn as nn

class HierarchicalGNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, g, node_feat, global_feat):
        return self.linear(global_feat[:, :1])
