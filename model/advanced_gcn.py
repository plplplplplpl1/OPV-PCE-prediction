"""
HighPCERegressorV3 — PyG implementation (used in all training scripts)

Three-branch GNN (GCN + GAT + GraphSAGE) with multi-scale pooling
and Morgan fingerprint fusion.

This is the canonical model definition. Training scripts should import
from here instead of redefining the class inline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool,
)


class HighPCERegressorV3(nn.Module):
    """Three-branch GNN (GCN+GAT+GraphSAGE) + Morgan fingerprint fusion."""

    def __init__(self, in_channels=30, hidden=128, fp_dim=512, dropout=0.3):
        super().__init__()
        # GCN branch
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        # GAT branch
        self.gat1 = GATConv(in_channels, hidden // 4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        # SAGE branch
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Morgan fingerprint encoder
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Fused dim: 3 branches x 3 pools x hidden + 128 (fp)
        fused_dim = hidden * 9 + 128

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))

        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))

        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))

        def pool3(h):
            return torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch),
            ], dim=1)

        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(data.fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)


# Alias for backward compatibility with saved checkpoints and docs
AdvancedGCN = HighPCERegressorV3
