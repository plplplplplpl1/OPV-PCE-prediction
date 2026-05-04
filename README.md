# OPV-PCE-Prediction

**Systematic Comparison of Tree Models (XGBoost) and Graph Neural Networks for Organic Photovoltaic Efficiency Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)

This repository contains the complete reproduction code for the manuscript:

> **"Model Advantage Crosses with Data Scale: A Systematic Comparison of XGBoost and Graph Neural Networks in Organic Photovoltaic Efficiency Prediction"**

We systematically compare XGBoost (with Morgan fingerprints + RDKit descriptors) against multi-branch GNNs across **7 independent datasets**.

---

## Core Finding

XGBoost vs GNN relative performance is a continuous function of data size, with a crossover point where advantage reverses. Below the crossover, fingerprint-based tree models dominate; above it, GNNs prevail.

### Validated Across 7 Datasets

| Dataset | Domain | Size | XGBoost R2 | GNN R2 | Crossover |
|---------|--------|------|------------|--------|-----------|
| OPV (this study) | Experimental PCE | 3,018 | **0.736** | 0.635 | Not reached |
| CEPDB | Computed OPV PCE | 25,000 | 0.858 | **0.925** | ~1,549 |
| ESOL | Water solubility | 1,128 | 0.676 | **0.870** | ~382 |
| FreeSolv | Hydration free energy | 642 | 0.731 | **0.871** | ~188 |
| Lipophilicity | Lipophilicity | 4,200 | 0.505 | **0.658** | ~209 |
| QM9 | HOMO-LUMO gap | 133,885 | 0.904 | **0.950** | ~2,080 |
| NREL OPV | DFT HOMO-LUMO gap | 95,004 | 0.832 | 0.833 | Convergence |

### Key Results on OPV Dataset (PCE > 3%, N = 1,916)

| Model | R2 | MAE (%) | Train Time |
|-------|-----|---------|-----------|
| XGBoost (Optuna) | **0.7360** | **1.391** | 3.5 s |
| XGBoost (4-seed mean) | 0.686 +- 0.026 | 1.428 +- 0.051 | 3.5 s |
| HighPCERegressorV3 (4-seed) | 0.635 +- 0.039 | 1.548 +- 0.049 | ~40 min |
| GIN + FP (4-seed) | **0.814 +- 0.025** | **0.981 +- 0.067** | ~60 min |
| PNA + FP (4-seed) | **0.811 +- 0.025** | **0.975 +- 0.081** | ~80 min |

---

## Repository Structure

```
OPV-PCE-prediction/
  model/                     -- GNN model definitions
    advanced_gcn.py          -- HighPCERegressorV3 (PyTorch Geometric)
  experiments/               -- 35 experiment scripts
    run_baseline_models.py   -- Tree/linear baselines
    run_feature_ablation.py  -- Feature contribution analysis
    run_fingerprint_sensitivity.py -- Fingerprint type comparison
    run_nonparametric_bootstrap.py -- Bootstrap significance tests
    run_end_to_end_pipeline.py -- Pipeline evaluation
    run_molenet_cross_validation.py -- MoleculeNet validation
    run_qm9_scale_experiment.py -- QM9 scaling
    run_nrel_opv_validation.py -- NREL OPV validation
    run_cepdb_gnn_multiseed.py -- CEPDB multi-seed GNN
    cepdb_crossover_bootstrap.py -- CEPDB crossover estimation
    run_ssl_pretrain.py      -- Self-supervised pre-training
    run_pretrain_finetune.py -- Transfer learning
    ... (21 more scripts)
  external_validation/       -- Cross-dataset validation
  training/                  -- Core training scripts (29 files)
    train_high_pce_v3.py     -- Train HighPCERegressorV3
    train_high_pce_xgb.py    -- Train XGBoost
    train_gps_baseline.py    -- Train GraphGPS
  utils/                     -- Evaluation, plotting, preprocessing
  figures/                   -- Figure generation (12 scripts)
  data/                      -- Dataset documentation
  run_all.sh                 -- Full reproduction pipeline
  requirements.txt
  LICENSE                    -- MIT License
```

103 Python files across all directories.

---

## Requirements and Installation

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12+ | conda recommended |
| PyTorch | >= 2.1.0 | adjust CUDA version |
| torch-geometric | >= 2.4.0 | pip install |
| XGBoost | >= 2.0.0 | pip install |
| scikit-learn | >= 1.1.1 | pip install |
| Optuna | >= 3.0.0 | pip install |
| RDKit | latest | conda-forge ONLY |
| pandas | >= 1.4.2 | pip install |

```bash
conda create -n opv python=3.12 -y
conda activate opv
conda install -c conda-forge rdkit -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric
pip install xgboost scikit-learn optuna pandas numpy matplotlib seaborn scipy pyyaml tqdm
```

Verify:
```bash
python -c "import torch, torch_geometric, xgboost, rdkit; print('All imports OK')"
```

---

## Data Preparation

CSV columns in order: Name, Code, PCE(%), ..., SMILES (last column).

```bash
cp your_data.csv data/data.csv
```

Statistics: 3,018 total (3,011 valid), PCE 0.000002% to 18.77%, High-PCE (>3%): 1,916 (63.5%).

External datasets are downloaded programmatically.

---

## Quick Start

### Train XGBoost (3.5 s)

```bash
python training/train_high_pce_xgb.py --fp-bits 4096 --all-desc --save-best
```

Expected R2 ~ 0.730-0.740.

### Train GNN (40 min on GPU)

```bash
python training/train_high_pce_v3.py --seed 9999
```

Expected R2 ~ 0.630-0.640.

### Run Key Experiments

```bash
python experiments/run_baseline_models.py
python experiments/run_feature_ablation.py
python experiments/run_fingerprint_sensitivity.py
python experiments/run_nonparametric_bootstrap.py
```

### Full Reproduction (8-10 hours)

```bash
bash run_all.sh
```

---

## Experiment to Manuscript Mapping

| Section | Script | Hardware |
|---------|--------|----------|
| 2.2 | run_end_to_end_pipeline.py | GPU |
| 2.4 | run_fp_only_classification.py, run_xgb_classifier.py | CPU |
| 2.5 | training/train_high_pce_v3.py, train_high_pce_xgb.py | GPU/CPU |
| 2.6.1 | run_feature_ablation.py | CPU |
| 2.6.2 | run_embedding_reverse_ablation.py, run_pure_gnn_ablation.py | GPU |
| 2.6.3 | run_fingerprint_sensitivity.py | CPU |
| 2.6.4 | run_gnn_capacity_scan.py | GPU |
| 2.7 | training/learning_curve_power_law.py | CPU |
| 2.8 | run_residual_analysis.py | CPU |
| 2.10.1 | external_validation/external_validation_cepdb.py | GPU |
| 2.10.2 | external_validation/external_validation_hopv15.py | GPU |
| 2.10.3 | run_molenet_cross_validation.py | GPU |
| 2.10.4 | run_qm9_scale_experiment.py | GPU |
| 2.10.5 | run_nrel_opv_validation.py | GPU |
| Bootstrap | run_nonparametric_bootstrap.py | CPU |
| Uncertainty | run_uncertainty.py | GPU |
| Screening | run_screening_simulation.py | CPU |

---

## Model Architecture

### HighPCERegressorV3 (model/advanced_gcn.py)

Three-branch GNN with fingerprint fusion:
- GCN: 2 layers, 128 hidden
- GAT: 2 layers, 128 hidden, 4 heads
- GraphSAGE: 2 layers, 128 hidden
- Multi-scale pooling: mean + max + sum
- Morgan FP: 512-bit to 128-d via 2-layer MLP
- Regressor: 1280 to 256 to 64 to 1 (PCE)

Params: ~592K. AdamW. Huber loss. Early stopping at 20 epochs.

### XGBoost

- 4096-bit Morgan (radius=2) + 217 RDKit descriptors
- Optuna: lr=0.0117, max_depth=6, subsample=0.595, colsample_bytree=0.626
- 8000 rounds, early stopping at 300

---

## Datasets

| Dataset | Samples | Target | Source |
|---------|---------|--------|--------|
| OPV | 3,018 | Experimental PCE | Literature |
| CEPDB | 25,000 | Computed PCE | Harvard |
| HOPV15 | 350 | Experimental PCE | Harvard |
| ESOL | 1,128 | Water solubility | MoleculeNet |
| FreeSolv | 642 | Hydration free energy | MoleculeNet |
| Lipophilicity | 4,200 | log D | MoleculeNet |
| QM9 | 133,885 | HOMO-LUMO gap | QM9 |
| NREL OPV | 95,004 | HOMO-LUMO gap | NREL |

---

## Results Reference

| Experiment | Expected R2 |
|-----------|-------------|
| XGBoost (Optuna) | 0.730-0.740 |
| XGBoost (default) | 0.680-0.690 |
| GNN V3 | 0.630-0.640 |
| MACCS keys | 0.640-0.650 |
| Linear regression | < 0 |

---

## Citation

```bibtex
@article{opv_pce_crossover_2026,
  title={Model Advantage Crosses with Data Scale: A Systematic Comparison of XGBoost and Graph Neural Networks in Organic Photovoltaic Efficiency Prediction},
  year={2026}
}
```

---

## License

MIT License - see LICENSE file.

---

## Notes

- Core model: model/advanced_gcn.py. Other model/ files are stubs.
- Random seeds: 42, 123, 333, 9999.
- GNN needs GPU with >= 8 GB VRAM.
- QM9/NREL OPV (n=50k) need >= 16 GB GPU VRAM.
- Full dataset must be obtained separately.
