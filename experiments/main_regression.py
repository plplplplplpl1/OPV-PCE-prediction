"""
Main Training Script for OPV Molecular PCE Regression

This script implements the complete training pipeline for predicting OPV power
conversion efficiency (PCE) using KA-GNN.

Key modifications from classification:
- Single-value regression instead of multi-label classification
- No label masking (all samples have PCE values)
- Different loss functions (Huber, MSE, MAE)
- Different evaluation metrics (MAE, RMSE, R2, Pearson)
"""

import os
import sys
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import dgl
import pickle

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.ka_gnn_regression import KA_GNN_Regression
from model.ensemble_model import EnsembleGNN
from model.ensemble_model_with_code import EnsembleGNN_WithCode
from model.hierarchical_model import HierarchicalGNN
from utils.graph_path import path_complex_mol
from utils.graph_3d import atom_to_graph_3d_with_fallback
from utils.splitters import ScaffoldSplitter, RandomSplitter
from utils.data_augmentation import SmilesAugmenter, randomize_smiles

# Load Code prefix mapping
CODE_PREFIX_MAPPING = None
try:
    with open('data/code_prefix_mapping.pkl', 'rb') as f:
        CODE_PREFIX_MAPPING = pickle.load(f)
    print(f"[INFO] Loaded Code prefix mapping with {CODE_PREFIX_MAPPING['n_classes']} classes")
except FileNotFoundError:
    print("[WARNING] Code prefix mapping not found. Will use only original global features.")
except Exception as e:
    print(f"[WARNING] Error loading Code prefix mapping: {e}")


# ================================================================
# Utility Functions
# ================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ================================================================
# Dataset and DataLoader
# ================================================================

class MolecularDataset(Dataset):
    """Custom dataset for molecular graphs"""
    def __init__(self, labels, graphs, smiles=None):
        self.labels = labels
        self.graphs = graphs
        self.smiles = smiles if smiles is not None else [None] * len(labels)

        # Get code indices if mapping is available
        self.code_indices = []
        if CODE_PREFIX_MAPPING is not None:
            for smi in self.smiles:
                if smi and smi in CODE_PREFIX_MAPPING['smiles_to_code']:
                    self.code_indices.append(CODE_PREFIX_MAPPING['smiles_to_code'][smi])
                else:
                    self.code_indices.append(0)  # Default to class 0 if not found
        else:
            self.code_indices = [0] * len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.graphs[idx], self.code_indices[idx]


def collate_fn(batch):
    """Custom collate function for batching graphs"""
    labels, graphs, code_indices = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.float32)
    code_indices = torch.tensor(code_indices, dtype=torch.long)
    batched_graph = dgl.batch(graphs)
    return labels, batched_graph, code_indices


def update_node_features(graph):
    """
    Update node features by aggregating edge information
    This enriches node representations with local structural context
    """
    # Check if features need to be updated (only if not already updated)
    if graph.ndata['feat'].shape[1] == 92:  # Original feature dimension
        graph.update_all(
            message_func=dgl.function.copy_e('feat', 'm'),
            reduce_func=dgl.function.mean('m', 'agg_feat')
        )
        # Concatenate original node features with aggregated edge features
        graph.ndata['feat'] = torch.cat([graph.ndata['feat'], graph.ndata['agg_feat']], dim=1)
    return graph


def extract_global_features(batched_graph):
    """
    Extract global features from a batched graph

    Args:
        batched_graph: Batched DGL graph

    Returns:
        Tensor of global features with shape (batch_size, 9)
    """
    # Get the list of individual graphs
    graphs = dgl.unbatch(batched_graph)

    # Extract global features from each graph
    global_feats = []
    for g in graphs:
        if hasattr(g, 'global_feat'):
            global_feats.append(g.global_feat)
        else:
            # If no global features, use zeros
            global_feats.append(torch.zeros(9))

    # Stack into a batch
    global_feats = torch.stack(global_feats)

    return global_feats


# ================================================================
# Data Loading and Preprocessing
# ================================================================

def create_data(config):
    """
    Create and preprocess molecular graph dataset

    Args:
        config: Configuration dictionary

    Returns:
        train_loader, valid_loader, test_loader
    """
    dataset_config = config['dataset']
    model_config = config['model']
    split_config = config['data_split']
    training_config = config['training']

    input_file = dataset_config['input_file']
    encoder_atom = model_config['encoder_atom']
    encoder_bond = model_config['encoder_bond']

    print("=" * 60)
    print("Creating Molecular Graph Dataset")
    print("=" * 60)

    # Check if processed data exists
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Use merged dataset processed file if using merged data
    if 'merged' in dataset_config.get('input_file', ''):
        processed_file_merged = os.path.join(processed_dir, 'merged_data_graphs.pth')
        processed_file_original = os.path.join(processed_dir, 'merged_data_graphs_original.pth')
    else:
        processed_file_merged = os.path.join(processed_dir, 'opv_graphs.pth')
        processed_file_original = os.path.join(processed_dir, 'opv_graphs_original.pth')

    # Determine which file to load based on configuration
    augment_config = config.get('augmentation', {})
    use_preconverted_augmented = augment_config.get('use_preconverted_augmented', False)

    if use_preconverted_augmented:
        # Use pre-converted augmented data (fastest)
        processed_file = processed_file_merged
        file_description = "pre-converted augmented data"
    else:
        # Use original data only
        processed_file = processed_file_original
        file_description = "original data (no augmentation)"

    # Fallback if file doesn't exist
    if not os.path.exists(processed_file):
        processed_file = processed_file_merged
        file_description = "processed data (fallback)"

    if os.path.exists(processed_file):
        print(f"\n[*] Loading {file_description} from {processed_file}")
        data = torch.load(processed_file)
        smiles_list = data['smiles']
        labels = data['labels']
        graphs = data['graphs']
        print(f"    Loaded {len(graphs)} molecular graphs")

        # Display data info
        if 'graph_type_stats' in data:
            stats = data.get('graph_type_stats', {})
            if isinstance(stats, dict):
                # Check if it's combined stats (has 'total' key)
                if 'total' in stats:
                    print(f"    Data contains: original + augmented graphs")
                    total_stats = stats['total']
                else:
                    total_stats = stats

                # Display graph type distribution
                total_count = sum([v for k, v in total_stats.items() if k != 'FAILED'])
                if total_count > 0:
                    print(f"    Graph type distribution:")
                    for key in ['3D_MMFF', '3D_UFF', '2D_FALLBACK']:
                        if key in total_stats:
                            count = total_stats[key]
                            print(f"      {key:15s}: {count:5d} ({count/total_count*100:.1f}%)")

        # Override runtime augmentation if using pre-converted augmented data
        if use_preconverted_augmented:
            print(f"    Using pre-converted augmented data - runtime augmentation disabled")
            runtime_augmentation_enabled = False
        else:
            runtime_augmentation_enabled = augment_config.get('enable', False)
    else:
        print(f"\n[1/4] Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        smiles_list = df['smiles'].tolist()
        labels = df['label'].values

        print(f"    Total samples: {len(smiles_list)}")
        print(f"    Label range: [{labels.min():.4f}, {labels.max():.4f}]")

        # Convert SMILES to molecular graphs
        use_3d = model_config.get('use_3d', True)

        print(f"\n[2/4] Converting SMILES to molecular graphs...")
        print(f"    Encoder: {encoder_atom} (atom), {encoder_bond} (bond)")
        if use_3d:
            print(f"    Using 3D structure (MMFF -> UFF -> 2D fallback)")
        else:
            print(f"    Using 2D structure only")

        graphs = []
        valid_indices = []
        failed_indices = []
        failed_reasons = []
        graph_type_stats = {'3D_MMFF': 0, '3D_UFF': 0, '2D_FALLBACK': 0}

        for i, smi in enumerate(smiles_list):
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{len(smiles_list)}")

            if use_3d:
                result = atom_to_graph_3d_with_fallback(smi, encoder_atom, encoder_bond)
            else:
                result = path_complex_mol(smi, encoder_atom, encoder_bond)

            # Check if result is a tuple (error case) or a graph (success)
            if isinstance(result, tuple):
                g, error_msg = result
                failed_indices.append(i)
                failed_reasons.append(error_msg)
                smi_short = smi[:80] + "..." if len(smi) > 80 else smi
                print(f"    [X] Index {i} failed: {error_msg}")
                print(f"        SMILES: {smi_short}")
            elif result is not False:
                graphs.append(result)
                valid_indices.append(i)
                # Track graph type
                if use_3d and hasattr(result, 'graph_type'):
                    graph_type_stats[result.graph_type] = graph_type_stats.get(result.graph_type, 0) + 1
            else:
                # Backward compatibility: old-style False return
                failed_indices.append(i)
                failed_reasons.append("Unknown error (legacy return)")
                print(f"    [X] Index {i} failed: Unknown error")

        # Filter out failed conversions
        labels = labels[valid_indices]
        smiles_list = [smiles_list[i] for i in valid_indices]

        # Print conversion summary
        failed_count = len(failed_indices)
        success_count = len(graphs)
        total_count = success_count + failed_count

        print(f"\n    Conversion Summary:")
        print(f"      Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"      Failed:  {failed_count}/{total_count} ({failed_count/total_count*100:.1f}%)")

        if use_3d and sum(graph_type_stats.values()) > 0:
            total_graphs = sum(graph_type_stats.values())
            print(f"\n    Graph Type Distribution:")
            print(f"      3D (MMFF):     {graph_type_stats.get('3D_MMFF', 0)} ({graph_type_stats.get('3D_MMFF', 0)/total_graphs*100:.1f}%)")
            print(f"      3D (UFF):      {graph_type_stats.get('3D_UFF', 0)} ({graph_type_stats.get('3D_UFF', 0)/total_graphs*100:.1f}%)")
            print(f"      2D (Fallback): {graph_type_stats.get('2D_FALLBACK', 0)} ({graph_type_stats.get('2D_FALLBACK', 0)/total_graphs*100:.1f}%)")
            total_3d = graph_type_stats.get('3D_MMFF', 0) + graph_type_stats.get('3D_UFF', 0)
            print(f"      Total 3D:      {total_3d} ({total_3d/total_graphs*100:.1f}%)")

        # Save failed molecules report if there are failures
        if failed_indices:
            os.makedirs('results', exist_ok=True)
            failed_df = pd.DataFrame({
                'index': failed_indices,
                'smiles': [smiles_list[i] if i < len(smiles_list) else df['smiles'].iloc[i] for i in failed_indices],
                'reason': failed_reasons
            })
            failed_df.to_csv('results/failed_molecules.csv', index=False)
            print(f"      Failed details saved to results/failed_molecules.csv")

        # Save processed data
        print(f"\n[3/4] Saving processed data to {processed_file}...")
        torch.save({
            'smiles': smiles_list,
            'labels': labels,
            'graphs': graphs
        }, processed_file)
        print(f"    Saved successfully")

        # Set runtime augmentation flag for freshly converted data
        runtime_augmentation_enabled = augment_config.get('enable', False)

    # Split data
    print(f"\n[4/4] Splitting data...")
    print(f"    Method: {split_config['method']}")
    print(f"    Ratios: train={split_config['train_ratio']}, "
          f"valid={split_config['valid_ratio']}, test={split_config['test_ratio']}")

    dataset = list(zip(smiles_list, labels, graphs))

    if split_config['method'] == 'scaffold':
        # Use scaffold splitting for chemical diversity
        splitter = ScaffoldSplitter()
        # ScaffoldSplitter expects dataset items to have SMILES as first element
        dataset_for_split = [(smi, label, graph) for smi, label, graph in dataset]
        train_data, valid_data, test_data = splitter.split(
            dataset_for_split,
            frac_train=split_config['train_ratio'],
            frac_valid=split_config['valid_ratio'],
            frac_test=split_config['test_ratio']
        )
    elif split_config['method'] == 'random':
        splitter = RandomSplitter()
        train_data, valid_data, test_data = splitter.split(
            dataset,
            frac_train=split_config['train_ratio'],
            frac_valid=split_config['valid_ratio'],
            frac_test=split_config['test_ratio'],
            seed=split_config['seed']
        )
    else:
        raise ValueError(f"Invalid split method: {split_config['method']}")

    # Extract labels and graphs
    train_smiles = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]
    train_graphs = [item[2] for item in train_data]

    valid_smiles = [item[0] for item in valid_data]
    valid_labels = [item[1] for item in valid_data]
    valid_graphs = [item[2] for item in valid_data]

    test_smiles = [item[0] for item in test_data]
    test_labels = [item[1] for item in test_data]
    test_graphs = [item[2] for item in test_data]

    print(f"    Train: {len(train_labels)} samples")
    print(f"    Valid: {len(valid_labels)} samples")
    print(f"    Test:  {len(test_labels)} samples")

    # Apply SMILES augmentation to training set only (if not using pre-converted augmented data)
    if runtime_augmentation_enabled:
        print(f"\n[Data Augmentation] Runtime SMILES randomization enabled")
        n_augment = augment_config.get('n_augment', 3)
        print(f"    Generating {n_augment} variants per molecule (on-the-fly)...")

        augmented_train_smiles = []
        augmented_train_labels = []

        for smiles, label in zip(train_smiles, train_labels):
            # Generate randomized SMILES
            augmented_train_smiles.append(smiles)  # Keep original
            augmented_train_labels.append(label)

            for _ in range(n_augment):
                aug_smiles = randomize_smiles(smiles, random_type="restricted")
                augmented_train_smiles.append(aug_smiles)
                augmented_train_labels.append(label)

        # Convert augmented SMILES to graphs (with 3D)
        use_3d = config.get('model', {}).get('use_3d', True)

        if use_3d:
            print(f"    Converting {len(augmented_train_smiles)} augmented SMILES to 3D graphs...")
            print(f"    (Fallback to 2D if 3D fails)")
        else:
            print(f"    Converting {len(augmented_train_smiles)} augmented SMILES to 2D graphs...")

        augmented_train_graphs = []
        failed_aug = 0
        graph_type_stats = {'3D_MMFF': 0, '3D_UFF': 0, '2D_FALLBACK': 0}

        for i, smi in enumerate(augmented_train_smiles):
            if (i + 1) % 500 == 0:
                print(f"      Progress: {i+1}/{len(augmented_train_smiles)}")

            if use_3d:
                result = atom_to_graph_3d_with_fallback(smi, encoder_atom, encoder_bond)
            else:
                result = path_complex_mol(smi, encoder_atom, encoder_bond)

            if isinstance(result, tuple) or result is False:
                # Augmented SMILES failed, skip this variant
                failed_aug += 1
            else:
                augmented_train_graphs.append(result)
                # Track graph type
                if hasattr(result, 'graph_type'):
                    graph_type_stats[result.graph_type] = graph_type_stats.get(result.graph_type, 0) + 1

        # Filter out failed augmentations
        if failed_aug > 0:
            print(f"    Warning: {failed_aug} augmented SMILES failed conversion")
            # Match labels to successful graphs
            augmented_train_labels = augmented_train_labels[:len(augmented_train_graphs)]

        train_labels = augmented_train_labels
        train_graphs = augmented_train_graphs

        print(f"    [OK] Augmented training set: {len(train_graphs)} samples "
              f"({len(train_graphs) - len(train_smiles)} added)")

        if use_3d and sum(graph_type_stats.values()) > 0:
            total = sum(graph_type_stats.values())
            print(f"    Graph type distribution:")
            print(f"      3D (MMFF):  {graph_type_stats.get('3D_MMFF', 0)} ({graph_type_stats.get('3D_MMFF', 0)/total*100:.1f}%)")
            print(f"      3D (UFF):   {graph_type_stats.get('3D_UFF', 0)} ({graph_type_stats.get('3D_UFF', 0)/total*100:.1f}%)")
            print(f"      2D (Fallback): {graph_type_stats.get('2D_FALLBACK', 0)} ({graph_type_stats.get('2D_FALLBACK', 0)/total*100:.1f}%)")

    # Create datasets
    train_dataset = MolecularDataset(train_labels, train_graphs, train_smiles)
    valid_dataset = MolecularDataset(valid_labels, valid_graphs, valid_smiles)
    test_dataset = MolecularDataset(test_labels, test_graphs, test_smiles)

    # Create dataloaders
    batch_size = training_config['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("\n" + "=" * 60)

    return train_loader, valid_loader, test_loader, train_graphs


# ================================================================
# Training and Evaluation
# ================================================================

def train_one_epoch(model, device, train_loader, optimizer, loss_fn, grad_clip=None, use_code=False, use_hierarchical=False, config=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_data in train_loader:
        # Always unpack 3 values since collate_fn returns (labels, graphs, code_indices)
        labels, graphs, code_indices = batch_data
        if use_code:
            code_indices = code_indices.to(device)
        else:
            code_indices = None

        labels = labels.to(device)
        graphs = update_node_features(graphs).to(device)
        features = graphs.ndata['feat']

        # Extract global features
        global_feats = extract_global_features(graphs).to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_hierarchical:
            # Hierarchical model returns (predictions, class_logits, class_labels) in training mode
            predictions, class_logits, class_labels = model(graphs, features, global_feats, mode='train', true_labels=labels)

            # Calculate losses
            classification_criterion = nn.CrossEntropyLoss()
            class_loss = classification_criterion(class_logits, class_labels)
            reg_loss = loss_fn(predictions.squeeze(-1), labels)

            # Combined loss
            classification_weight = config.get('training', {}).get('classification_weight', 0.3)
            regression_weight = config.get('training', {}).get('regression_weight', 0.7)
            loss = classification_weight * class_loss + regression_weight * reg_loss
        elif use_code:
            outputs = model(graphs, features, global_feats, code_indices)
            loss = loss_fn(outputs.squeeze(-1), labels)
        else:
            outputs = model(graphs, features, global_feats)
            loss = loss_fn(outputs.squeeze(-1), labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, device, data_loader, loss_fn, use_code=False, use_hierarchical=False):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            # Always unpack 3 values since collate_fn returns (labels, graphs, code_indices)
            labels, graphs, code_indices = batch_data
            if use_code:
                code_indices = code_indices.to(device)
            else:
                code_indices = None

            labels = labels.to(device)
            graphs = update_node_features(graphs).to(device)
            features = graphs.ndata['feat']

            # Extract global features
            global_feats = extract_global_features(graphs).to(device)

            # Forward pass
            if use_hierarchical:
                # Hierarchical model in test mode returns only predictions
                outputs = model(graphs, features, global_feats, mode='test')
            elif use_code:
                outputs = model(graphs, features, global_feats, code_indices)
            else:
                outputs = model(graphs, features, global_feats)

            # Calculate loss
            loss = loss_fn(outputs.squeeze(-1), labels)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and labels
            preds = outputs.squeeze(-1).cpu().numpy()
            if preds.ndim == 0:  # Handle single sample case
                preds = [preds.item()]
            else:
                preds = preds.tolist()
            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / num_batches
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    return avg_loss, predictions, labels


def calculate_metrics(predictions, labels, norm_params=None):
    """
    Calculate regression metrics

    Args:
        predictions: Predicted PCE values (may be normalized)
        labels: True PCE values (may be normalized)
        norm_params: Normalization parameters for inverse transform

    Returns:
        Dictionary of metrics
    """
    # Check for NaN values
    if np.isnan(predictions).any():
        print(f"  Warning: Predictions contain {np.isnan(predictions).sum()} NaN values")
        predictions = np.nan_to_num(predictions, nan=0.0)

    if np.isnan(labels).any():
        print(f"  Warning: Labels contain {np.isnan(labels).sum()} NaN values")
        labels = np.nan_to_num(labels, nan=0.0)

    # Inverse transform if normalization was applied
    if norm_params is not None and norm_params.get('method') == 'minmax':
        pce_min = norm_params['min']
        pce_max = norm_params['max']
        # Denormalize: original = normalized * (max - min) + min
        pred_values = predictions * (pce_max - pce_min) + pce_min
        true_values = labels * (pce_max - pce_min) + pce_min
    else:
        # No transformation needed
        pred_values = predictions
        true_values = labels

    # Calculate metrics on original PCE values
    mae = mean_absolute_error(true_values, pred_values)
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    r2 = r2_score(true_values, pred_values)

    # Pearson and Spearman correlations
    pearson, _ = pearsonr(true_values, pred_values)
    spearman, _ = spearmanr(true_values, pred_values)

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-6))) * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
        'mape': mape
    }

    return metrics


# ================================================================
# Main Training Loop
# ================================================================

def main(config_path='config/opv_regression.yaml'):
    """Main training function"""

    # Load configuration
    config = load_config(config_path)

    # Set random seed
    set_seed(config['data_split']['seed'])

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"\nDevice: {device}")

    # Create data loaders
    train_loader, valid_loader, test_loader, train_graphs = create_data(config)

    # Load normalization parameters
    try:
        norm_params = np.load('data/norm_params_merged.npy', allow_pickle=True).item()
        print("[INFO] Loaded merged normalization parameters")
    except FileNotFoundError:
        norm_params = np.load('data/norm_params.npy', allow_pickle=True).item()
        print("[INFO] Loaded original normalization parameters")

    # Model configuration
    model_config = config['model']

    # Calculate input feature dimension
    # Get actual edge feature dimension from first graph
    sample_graph = train_graphs[0]
    edge_feat_dim = sample_graph.edata['feat'].shape[1] if 'feat' in sample_graph.edata else 0
    in_feat = model_config['in_feat'] + edge_feat_dim  # After edge aggregation

    print(f"Node feature dim: {model_config['in_feat']}")
    print(f"Edge feature dim: {edge_feat_dim}")
    print(f"Total input dim: {in_feat}")

    # Create model based on configuration
    model_name = model_config.get('name', 'ka_gnn_regression')

    if model_name == 'ensemble_gnn':
        # Ensemble model
        if CODE_PREFIX_MAPPING is not None:
            # Use ensemble with Code feature MLP
            model = EnsembleGNN_WithCode(
                in_feat=in_feat,
                hidden_feat=model_config['hidden_feat'],
                out_feat=model_config['out_feat'],
                grid_feat=model_config['grid_feat'],
                num_layers=model_config['num_layers'],
                num_heads=model_config.get('num_heads', 4),
                dropout=model_config['dropout'],
                pooling=model_config['pooling'],
                global_feat_dim=9,  # Keep original 9-dim global features
                n_code_classes=CODE_PREFIX_MAPPING['n_classes'],
                code_embed_dim=8,
                fusion_method=model_config.get('fusion_method', 'weighted')
            ).to(device)
            use_code_feature = True
            use_hierarchical = False
        else:
            # Standard ensemble without Code features
            model = EnsembleGNN(
                in_feat=in_feat,
                hidden_feat=model_config['hidden_feat'],
                out_feat=model_config['out_feat'],
                grid_feat=model_config['grid_feat'],
                num_layers=model_config['num_layers'],
                num_heads=model_config.get('num_heads', 4),
                dropout=model_config['dropout'],
                pooling=model_config['pooling'],
                global_feat_dim=9,
                fusion_method=model_config.get('fusion_method', 'weighted')
            ).to(device)
            use_code_feature = False
            use_hierarchical = False
        # Hierarchical two-stage model
        model = HierarchicalGNN(
            in_feat=in_feat,
            hidden_feat=model_config['hidden_feat'],
            out_feat=model_config['out_feat'],
            grid_feat=model_config['grid_feat'],
            num_layers=model_config['num_layers'],
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config['dropout'],
            pooling=model_config['pooling'],
            global_feat_dim=9,
            fusion_method=model_config.get('fusion_method', 'weighted'),
            classifier_hidden=model_config.get('classifier_hidden', 128),
            threshold=model_config.get('threshold', 3.0)
        ).to(device)
        use_code_feature = False
        use_hierarchical = True
    else:
        # Default KA-GNN model
        model = KA_GNN_Regression(
            in_feat=in_feat,
            hidden_feat=model_config['hidden_feat'],
            out_feat=model_config['out_feat'],
            grid_feat=model_config['grid_feat'],
            num_layers=model_config['num_layers'],
            pooling=model_config['pooling'],
            dropout=model_config['dropout'],
            use_bias=model_config['use_bias'],
            global_feat_dim=9
        ).to(device)
        use_code_feature = False
        use_hierarchical = False

    print(f"\nModel: {model_config['name']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    if use_hierarchical:
        print(f"Hierarchical model with threshold: {model_config.get('threshold', 3.0)}%")
    if CODE_PREFIX_MAPPING is not None and use_code_feature:
        print(f"Using Code features: {CODE_PREFIX_MAPPING['n_classes']} classes")

    # Loss function
    training_config = config['training']
    loss_type = training_config['loss']

    if loss_type == 'huber':
        loss_fn = nn.SmoothL1Loss(beta=training_config.get('huber_delta', 1.0))
    elif loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type == 'smooth_l1':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Invalid loss function: {loss_type}")

    print(f"Loss function: {loss_type}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config.get('weight_decay', 0.0)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=training_config.get('scheduler_patience', 10),
        factor=training_config.get('scheduler_factor', 0.5),
        min_lr=training_config.get('scheduler_min_lr', 1e-6)
    )

    # Training loop
    num_epochs = training_config['num_epochs']
    early_stopping_patience = training_config.get('early_stopping_patience', 30)
    best_valid_loss = float('inf')
    patience_counter = 0

    # Create checkpoint directory
    checkpoint_dir = config['experiment']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Training Started")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, device, train_loader, optimizer, loss_fn,
            grad_clip=training_config.get('grad_clip'),
            use_code=use_code_feature,
            use_hierarchical=use_hierarchical,
            config=config
        )

        # Validate
        valid_loss, valid_preds, valid_labels = evaluate(model, device, valid_loader, loss_fn, use_code=use_code_feature, use_hierarchical=use_hierarchical)

        # Calculate metrics
        valid_metrics = calculate_metrics(valid_preds, valid_labels, norm_params)

        # Learning rate scheduling
        scheduler.step(valid_loss)

        # Logging
        if epoch % config['logging']['log_interval'] == 0:
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {valid_loss:.4f}")
            print(f"  Valid MAE:  {valid_metrics['mae']:.4f}%")
            print(f"  Valid RMSE: {valid_metrics['rmse']:.4f}%")
            print(f"  Valid R2:   {valid_metrics['r2']:.4f}")
            print(f"  Valid Pearson: {valid_metrics['pearson']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping and checkpointing
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0

            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'valid_metrics': valid_metrics,
                'config': config
            }, checkpoint_path)

            print(f"  [OK] Best model saved (Valid Loss: {valid_loss:.4f})")
        else:
            patience_counter += 1

        if training_config.get('early_stopping', True) and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating Best Model on Test Set")
    print("=" * 60)

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_preds, test_labels = evaluate(model, device, test_loader, loss_fn, use_code=use_code_feature, use_hierarchical=use_hierarchical)
    test_metrics = calculate_metrics(test_preds, test_labels, norm_params)

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  MAE:       {test_metrics['mae']:.4f}%")
    print(f"  RMSE:      {test_metrics['rmse']:.4f}%")
    print(f"  R2:        {test_metrics['r2']:.4f}")
    print(f"  Pearson:   {test_metrics['pearson']:.4f}")
    print(f"  Spearman:  {test_metrics['spearman']:.4f}")
    print(f"  MAPE:      {test_metrics['mape']:.2f}%")

    # Denormalize predictions for saving
    if norm_params is not None and norm_params.get('method') == 'minmax':
        pce_min = norm_params['min']
        pce_max = norm_params['max']
        test_preds_original = test_preds * (pce_max - pce_min) + pce_min
        test_labels_original = test_labels * (pce_max - pce_min) + pce_min
    else:
        test_preds_original = test_preds
        test_labels_original = test_labels

    # Save test predictions in original PCE scale
    results_df = pd.DataFrame({
        'true_pce': test_labels_original,
        'pred_pce': test_preds_original,
        'absolute_error': np.abs(test_labels_original - test_preds_original),
        'relative_error': np.abs(test_labels_original - test_preds_original) / (test_labels_original + 1e-6) * 100
    })
    results_df.to_csv('results/test_predictions.csv', index=False)
    print(f"\n[OK] Test predictions saved to results/test_predictions.csv")

    print("\n" + "=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)

    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KA-GNN for OPV PCE Regression')
    parser.add_argument('--config', type=str, default='config/opv_regression.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    main(args.config)
