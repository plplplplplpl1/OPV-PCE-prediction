"""
Evaluation and Visualization Script for OPV PCE Regression

This script:
1. Loads a trained model
2. Evaluates it on test data
3. Generates comprehensive visualizations
4. Analyzes error patterns
5. Produces a detailed report
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from experiments.main_regression import create_data, evaluate, calculate_metrics
from model.ka_gnn_regression import KA_GNN_Regression
from model.ensemble_model import EnsembleGNN


def plot_prediction_results(y_true, y_pred, metrics, save_path='results/prediction_analysis.png'):
    """
    Visualize prediction results with multiple plots

    Args:
        y_true: True PCE values (denormalized)
        y_pred: Predicted PCE values (denormalized)
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: Scatter plot (Predicted vs True)
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')

    axes[0, 0].set_xlabel('True PCE (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted PCE (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_title(f'Prediction vs True Values\nR2 = {metrics["r2"]:.3f}, MAE = {metrics["mae"]:.3f}%',
                         fontsize=15, fontweight='bold')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(alpha=0.3)

    # Add text box with metrics
    textstr = f'RMSE = {metrics["rmse"]:.3f}%\nPearson = {metrics["pearson"]:.3f}\nSpearman = {metrics["spearman"]:.3f}'
    axes[0, 0].text(0.05, 0.95, textstr, transform=axes[0, 0].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Residual plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted PCE (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (True - Predicted)', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Residual Plot', fontsize=15, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Residual distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    axes[1, 0].axvline(np.mean(residuals), color='green', linestyle='--', lw=2,
                       label=f'Mean = {np.mean(residuals):.3f}')
    axes[1, 0].set_xlabel('Residuals (True - Predicted)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('Residual Distribution', fontsize=15, fontweight='bold')
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Plot 4: Error by PCE range
    ranges = ['0-2%', '2-4%', '4-6%', '6-8%', '8-10%', '>10%']
    range_masks = [
        (y_true >= 0) & (y_true < 2),
        (y_true >= 2) & (y_true < 4),
        (y_true >= 4) & (y_true < 6),
        (y_true >= 6) & (y_true < 8),
        (y_true >= 8) & (y_true < 10),
        y_true >= 10
    ]

    mae_by_range = []
    counts_by_range = []

    for mask in range_masks:
        if mask.sum() > 0:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            mae_by_range.append(mae)
            counts_by_range.append(mask.sum())
        else:
            mae_by_range.append(0)
            counts_by_range.append(0)

    colors = plt.cm.viridis(np.linspace(0, 1, len(ranges)))
    bars = axes[1, 1].bar(ranges, mae_by_range, edgecolor='black', alpha=0.7, color=colors)
    axes[1, 1].set_xlabel('PCE Range', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('MAE (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('MAE by PCE Range', fontsize=15, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')

    # Add count labels on bars
    for bar, count in zip(bars, counts_by_range):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'n={count}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved prediction analysis to {save_path}")
    plt.close()


def plot_error_analysis(y_true, y_pred, save_path='results/error_analysis.png'):
    """
    Detailed error analysis plots

    Args:
        y_true: True PCE values
        y_pred: Predicted PCE values
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    rel_errors = np.abs(residuals) / (y_true + 1e-6) * 100

    # Plot 1: Absolute error vs True PCE
    axes[0, 0].scatter(y_true, abs_errors, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    axes[0, 0].set_xlabel('True PCE (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 0].set_title('Absolute Error vs True PCE', fontsize=15, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Relative error distribution
    axes[0, 1].hist(rel_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_xlabel('Relative Error (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Relative Error Distribution', fontsize=15, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')

    # Plot 3: Q-Q plot for residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Residuals)', fontsize=15, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Cumulative error distribution
    sorted_abs_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100

    axes[1, 1].plot(sorted_abs_errors, cumulative, linewidth=2, color='darkblue')
    axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    axes[1, 1].axhline(90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    axes[1, 1].set_xlabel('Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('Cumulative Error Distribution', fontsize=15, fontweight='bold')
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved error analysis to {save_path}")
    plt.close()


def analyze_worst_predictions(y_true, y_pred, n=10, save_path='results/worst_predictions.txt'):
    """
    Analyze worst predictions

    Args:
        y_true: True PCE values
        y_pred: Predicted PCE values
        n: Number of worst predictions to analyze
        save_path: Path to save the analysis
    """
    abs_errors = np.abs(y_true - y_pred)
    worst_indices = np.argsort(abs_errors)[-n:][::-1]

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Top {n} Worst Predictions\n")
        f.write("=" * 80 + "\n\n")

        for rank, idx in enumerate(worst_indices, 1):
            f.write(f"Rank {rank}:\n")
            f.write(f"  Index:           {idx}\n")
            f.write(f"  True PCE:        {y_true[idx]:.4f}%\n")
            f.write(f"  Predicted PCE:   {y_pred[idx]:.4f}%\n")
            f.write(f"  Absolute Error:  {abs_errors[idx]:.4f}%\n")
            f.write(f"  Relative Error:  {abs_errors[idx]/(y_true[idx]+1e-6)*100:.2f}%\n")
            f.write("\n")

    print(f"[OK] Saved worst predictions analysis to {save_path}")


def generate_evaluation_report(metrics, save_path='results/evaluation_report.txt'):
    """
    Generate comprehensive evaluation report

    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OPV PCE Prediction - Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Regression Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean Absolute Error (MAE):           {metrics['mae']:.4f}%\n")
        f.write(f"  Root Mean Squared Error (RMSE):      {metrics['rmse']:.4f}%\n")
        f.write(f"  R-squared (R2):                      {metrics['r2']:.4f}\n")
        f.write(f"  Pearson Correlation:                 {metrics['pearson']:.4f}\n")
        f.write(f"  Spearman Correlation:                {metrics['spearman']:.4f}\n")
        f.write(f"  Mean Absolute Percentage Error:      {metrics['mape']:.2f}%\n")
        f.write("\n")

        f.write("Performance Assessment:\n")
        f.write("-" * 80 + "\n")

        # MAE assessment
        if metrics['mae'] < 0.8:
            mae_assessment = "Excellent"
        elif metrics['mae'] < 1.0:
            mae_assessment = "Very Good"
        elif metrics['mae'] < 1.5:
            mae_assessment = "Good"
        else:
            mae_assessment = "Needs Improvement"
        f.write(f"  MAE Assessment:  {mae_assessment}\n")

        # R2 assessment
        if metrics['r2'] > 0.90:
            r2_assessment = "Excellent"
        elif metrics['r2'] > 0.85:
            r2_assessment = "Very Good"
        elif metrics['r2'] > 0.75:
            r2_assessment = "Good"
        else:
            r2_assessment = "Needs Improvement"
        f.write(f"  R2 Assessment:   {r2_assessment}\n")

        # Pearson assessment
        if metrics['pearson'] > 0.90:
            pearson_assessment = "Excellent"
        elif metrics['pearson'] > 0.85:
            pearson_assessment = "Very Good"
        elif metrics['pearson'] > 0.75:
            pearson_assessment = "Good"
        else:
            pearson_assessment = "Needs Improvement"
        f.write(f"  Correlation:     {pearson_assessment}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[OK] Saved evaluation report to {save_path}")


def main(config_path='config/opv_regression.yaml', checkpoint_path='checkpoints/best_model.pth'):
    """Main evaluation function"""

    print("=" * 80)
    print("Model Evaluation and Analysis")
    print("=" * 80)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print(f"\n[1/5] Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = config['model']

    # Load processed data to get actual edge feature dimension
    data = torch.load('data/processed/opv_graphs.pth')
    sample_graph = data['graphs'][0]
    edge_feat_dim = sample_graph.edata['feat'].shape[1] if 'feat' in sample_graph.edata else 0
    in_feat = model_config['in_feat'] + edge_feat_dim  # After edge aggregation

    # Create model based on model name in config
    model_name = model_config.get('name', 'ka_gnn_regression')

    if model_name == 'ensemble_gnn':
        print(f"    Creating Ensemble GNN model...")
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
    else:
        print(f"    Creating KA-GNN model...")
        model = KA_GNN_Regression(
            in_feat=in_feat,
            hidden_feat=model_config['hidden_feat'],
            out_feat=model_config['out_feat'],
            grid_feat=model_config['grid_feat'],
            num_layers=model_config['num_layers'],
            pooling=model_config['pooling'],
            dropout=model_config['dropout'],
            use_bias=model_config['use_bias'],
            global_feat_dim=9  # 9 global molecular features
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Model loaded from epoch {checkpoint['epoch']}")

    # Create data loaders
    print(f"\n[2/5] Loading test data...")
    _, _, test_loader, _ = create_data(config)

    # Load normalization parameters
    norm_params = np.load('data/norm_params.npy', allow_pickle=True).item()

    # Evaluate on test set
    print(f"\n[3/5] Evaluating on test set...")
    loss_fn = torch.nn.SmoothL1Loss()  # Dummy loss function for evaluation
    test_loss, test_preds, test_labels = evaluate(model, device, test_loader, loss_fn)

    # Calculate metrics
    metrics = calculate_metrics(test_preds, test_labels, norm_params)

    print(f"\nTest Results:")
    print(f"  MAE:       {metrics['mae']:.4f}%")
    print(f"  RMSE:      {metrics['rmse']:.4f}%")
    print(f"  R2:        {metrics['r2']:.4f}")
    print(f"  Pearson:   {metrics['pearson']:.4f}")
    print(f"  Spearman:  {metrics['spearman']:.4f}")
    print(f"  MAPE:      {metrics['mape']:.2f}%")

    # Denormalize predictions
    if norm_params.get('method') == 'minmax':
        pce_min = norm_params['min']
        pce_max = norm_params['max']
        y_true = test_labels * (pce_max - pce_min) + pce_min
        y_pred = test_preds * (pce_max - pce_min) + pce_min
    else:
        y_true = test_labels
        y_pred = test_preds

    # Generate visualizations
    print(f"\n[4/5] Generating visualizations...")
    os.makedirs('results', exist_ok=True)

    plot_prediction_results(y_true, y_pred, metrics)
    plot_error_analysis(y_true, y_pred)

    # Analyze worst predictions
    print(f"\n[5/5] Analyzing predictions...")
    analyze_worst_predictions(y_true, y_pred, n=20)

    # Generate report
    generate_evaluation_report(metrics)

    print("\n" + "=" * 80)
    print("Evaluation Completed Successfully!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - results/prediction_analysis.png")
    print(f"  - results/error_analysis.png")
    print(f"  - results/worst_predictions.txt")
    print(f"  - results/evaluation_report.txt")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained KA-GNN model')
    parser.add_argument('--config', type=str, default='config/opv_regression.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    args = parser.parse_args()

    main(args.config, args.checkpoint)
