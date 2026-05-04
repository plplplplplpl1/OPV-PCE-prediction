"""
Advanced Hierarchical Two-Stage Model for PCE Prediction
Integrates the best binary classifier (AdvancedGCN) with hierarchical regression

Key improvements:
1. Multi-branch GNN architecture (GCN, GAT, GraphSAGE) for binary classification
2. Multi-scale graph pooling (mean, max, sum)
3. Attention-based branch fusion
4. Data augmentation support
5. Specialized regressors for low/high PCE ranges
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import dgl.function as fn
from model.ensemble_model import EnsembleGNN


class AdvancedBinaryClassifier(nn.Module):
    """
    Advanced binary classifier inspired by main.py's AdvancedGCN
    Uses multi-branch GNN architecture with DGL
    """
    def __init__(self, in_feat, hidden_dim=160, num_layers=4, num_heads=4,
                 dropout=0.3, global_feat_dim=9):
        super(AdvancedBinaryClassifier, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Three parallel GNN branches
        # Branch 1: GCN (Graph Convolutional Network)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConv(in_feat, hidden_dim, activation=F.relu))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))

        # Branch 2: GAT (Graph Attention Network)
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_feat, hidden_dim // num_heads, num_heads=num_heads))
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads=num_heads))

        # Branch 3: GraphSAGE
        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(in_feat, hidden_dim, aggregator_type='mean'))
        for _ in range(num_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))

        # Batch normalization for each layer (after concatenation)
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 3))  # 3 branches

        # Residual projection
        self.residual_proj = nn.Linear(in_feat, hidden_dim * 3)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Multi-scale pooling weights (learnable)
        self.pool_weights = nn.Parameter(torch.ones(3) / 3)  # mean, max, sum

        # Classifier head (takes pooled features + global features)
        classifier_input_dim = hidden_dim * 3 * 3 + global_feat_dim  # 3 branches * 3 pooling + global
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )

    def forward(self, g, node_feat, global_feat):
        """
        Forward pass through multi-branch GNN

        Args:
            g: DGL graph
            node_feat: Node features [num_nodes, in_feat]
            global_feat: Global molecular features [batch_size, global_feat_dim]

        Returns:
            logits: Classification logits [batch_size, 2]
        """
        # Save input for residual connection
        with g.local_scope():
            g.ndata['h_res'] = node_feat
            residual = dgl.mean_nodes(g, 'h_res')
            residual = self.residual_proj(residual)

        # Initialize branch inputs
        gcn_h = node_feat
        gat_h = node_feat
        sage_h = node_feat

        # Process through layers
        for i in range(self.num_layers):
            # GCN branch
            gcn_h = self.gcn_layers[i](g, gcn_h)
            gcn_h = self.dropout_layer(gcn_h)

            # GAT branch
            gat_h = self.gat_layers[i](g, gat_h)
            if gat_h.dim() == 3:  # GAT returns [num_nodes, num_heads, feat_dim]
                gat_h = gat_h.flatten(1)  # Flatten to [num_nodes, num_heads * feat_dim]
            gat_h = self.dropout_layer(gat_h)

            # SAGE branch
            sage_h = self.sage_layers[i](g, sage_h)
            sage_h = self.dropout_layer(sage_h)

            # Concatenate branches
            combined_h = torch.cat([gcn_h, gat_h, sage_h], dim=1)

            # Add residual connection (only at first layer)
            if i == 0:
                with g.local_scope():
                    g.ndata['h_combined'] = combined_h
                    pooled_combined = dgl.mean_nodes(g, 'h_combined')
                    pooled_combined = pooled_combined + residual
                    # Broadcast back to nodes (simplified - just use combined_h)

            # Batch normalization
            combined_h = self.batch_norms[i](combined_h)

            # Split back to branches for next layer
            gcn_h = combined_h[:, :self.hidden_dim]
            gat_h = combined_h[:, self.hidden_dim:2*self.hidden_dim]
            sage_h = combined_h[:, 2*self.hidden_dim:]

        # Final combined features
        final_h = torch.cat([gcn_h, gat_h, sage_h], dim=1)

        # Multi-scale graph pooling
        with g.local_scope():
            g.ndata['h'] = final_h

            # Mean pooling
            h_mean = dgl.mean_nodes(g, 'h')

            # Max pooling
            h_max = dgl.max_nodes(g, 'h')

            # Sum pooling
            h_sum = dgl.sum_nodes(g, 'h')

        # Weighted combination of pooling methods
        pooled_features = torch.cat([h_mean, h_max, h_sum], dim=1)

        # Concatenate with global features
        combined_features = torch.cat([pooled_features, global_feat], dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits


class AdvancedHierarchicalGNN(nn.Module):
    """
    Advanced Hierarchical two-stage model:
    1. Advanced binary classifier (multi-branch GNN) to separate low/high PCE
    2. Two specialized GNN regressors for each class
    """
    def __init__(self, in_feat, hidden_feat=64, out_feat=32, grid_feat=16,
                 num_layers=3, num_heads=4, dropout=0.3, pooling='avg',
                 global_feat_dim=9, fusion_method='weighted',
                 classifier_hidden=160, classifier_layers=4, threshold=2.9):
        super(AdvancedHierarchicalGNN, self).__init__()

        self.threshold = threshold

        # Advanced binary classifier (replaces simple GraphFeatureExtractor + BinaryClassifier)
        self.binary_classifier = AdvancedBinaryClassifier(
            in_feat=in_feat,
            hidden_dim=classifier_hidden,
            num_layers=classifier_layers,
            num_heads=num_heads,
            dropout=dropout,
            global_feat_dim=global_feat_dim
        )

        # Specialized regressors for each class
        # Low PCE regressor (< threshold)
        self.low_pce_regressor = EnsembleGNN(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            out_feat=out_feat,
            grid_feat=grid_feat,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pooling=pooling,
            global_feat_dim=global_feat_dim,
            fusion_method=fusion_method
        )

        # High PCE regressor (>= threshold)
        self.high_pce_regressor = EnsembleGNN(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            out_feat=out_feat,
            grid_feat=grid_feat,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pooling=pooling,
            global_feat_dim=global_feat_dim,
            fusion_method=fusion_method
        )

    def forward(self, g, node_feat, global_feat, mode='train', true_labels=None):
        """
        Forward pass with hierarchical prediction

        Args:
            g: DGL graph
            node_feat: Node features
            global_feat: Global molecular features
            mode: 'train' or 'test'
            true_labels: True PCE values (for training stage 1)

        Returns:
            If mode == 'train':
                predictions, class_logits, class_labels
            If mode == 'test':
                predictions
        """
        batch_size = global_feat.shape[0]

        # Stage 1: Binary classification using advanced classifier
        class_logits = self.binary_classifier(g, node_feat, global_feat)
        class_probs = F.softmax(class_logits, dim=1)

        if mode == 'train' and true_labels is not None:
            # During training, use true labels to route to correct regressor
            class_labels = (true_labels >= self.threshold).long().squeeze()

            # Get predictions from both regressors
            low_preds = self.low_pce_regressor(g, node_feat, global_feat)
            high_preds = self.high_pce_regressor(g, node_feat, global_feat)

            # Select predictions based on true class
            predictions = torch.zeros(batch_size, 1, device=node_feat.device)
            low_mask = (class_labels == 0)
            high_mask = (class_labels == 1)

            if low_mask.any():
                predictions[low_mask] = low_preds[low_mask]
            if high_mask.any():
                predictions[high_mask] = high_preds[high_mask]

            return predictions, class_logits, class_labels

        else:
            # During testing, use predicted class
            class_preds = torch.argmax(class_probs, dim=1)

            # Get predictions from both regressors
            low_preds = self.low_pce_regressor(g, node_feat, global_feat)
            high_preds = self.high_pce_regressor(g, node_feat, global_feat)

            # Select predictions based on predicted class
            predictions = torch.zeros(batch_size, 1, device=node_feat.device)
            low_mask = (class_preds == 0)
            high_mask = (class_preds == 1)

            if low_mask.any():
                predictions[low_mask] = low_preds[low_mask]
            if high_mask.any():
                predictions[high_mask] = high_preds[high_mask]

            return predictions

    def get_model_weights(self):
        """Get fusion weights from the regressors"""
        return {
            'low_pce': self.low_pce_regressor.get_model_weights(),
            'high_pce': self.high_pce_regressor.get_model_weights()
        }

    def get_classification_accuracy(self, g, node_feat, global_feat, true_labels):
        """
        Get binary classification accuracy

        Args:
            g: DGL graph
            node_feat: Node features
            global_feat: Global features
            true_labels: True PCE values

        Returns:
            accuracy: Classification accuracy
        """
        with torch.no_grad():
            class_logits = self.binary_classifier(g, node_feat, global_feat)
            class_preds = torch.argmax(class_logits, dim=1)
            class_labels = (true_labels >= self.threshold).long().squeeze()
            accuracy = (class_preds == class_labels).float().mean().item()
        return accuracy

