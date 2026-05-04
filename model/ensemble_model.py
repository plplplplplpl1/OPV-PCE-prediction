"""Stub module for compatibility with full-version model references.
The actual GNN model used in the paper is in advanced_gcn.py.
This stub allows older scripts (hierarchical.py, evaluate.py) to import without error.
"""
import torch.nn as nn


class EnsembleGNN(nn.Module):
    """Stub: full-version DGL ensemble model (not used in paper experiments)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, g, node_feat, global_feat):
        return self.linear(global_feat[:, :1])

    def get_model_weights(self):
        return {}


class KA_GNN_Regression(nn.Module):
    """Stub: full-version model (not used in paper experiments)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, g, node_feat, global_feat):
        return self.linear(global_feat[:, :1])
