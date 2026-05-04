#!/usr/bin/env python3
"""
XGBoost classifier for high/low PCE discrimination.
Uses the SAME 512-bit Morgan fingerprint input as the GNN classifier,
to test whether GNN's classification advantage is from richer input representation.
"""
import numpy as np
import pandas as pd
import os, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE_DIR = "/root/第四版r2=0.72/最小版本"
DATA_PATH = os.path.join(BASE_DIR, "data/data.csv")
PCE_THRESHOLD = 3.0

# ── Load data ──
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
for c in df.columns:
    if 'pce' in c.lower(): df = df.rename(columns={c: 'PCE'})
    if 'smiles' in c.lower(): df = df.rename(columns={c: 'SMILES'})
df = df.dropna(subset=['PCE', 'SMILES'])
print(f"Total samples: {len(df)}")
print(f"High PCE (>3%): {(df['PCE'] > PCE_THRESHOLD).sum()}")
print(f"Low PCE (≤3%): {(df['PCE'] <= PCE_THRESHOLD).sum()}")

# ── Feature: 512-bit Morgan fingerprint (matching GNN) ──
def featurize(smi, nbits=512, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)

X_list, y_list = [], []
failed = 0
for _, r in df.iterrows():
    f = featurize(r['SMILES'])
    if f is not None:
        X_list.append(f)
        y_list.append(1 if r['PCE'] > PCE_THRESHOLD else 0)
    else:
        failed += 1
X = np.array(X_list)
y = np.array(y_list)
print(f"Features: {X.shape[1]} (512-bit Morgan), failed SMILES: {failed}")

# ── Train XGBoost classifier ──
# Try two configurations:
# 1. Default parameters (for fair comparison with GNN which didn't optimize)
# 2. Optimized parameters (best effort)

configs = {
    'default': {
        'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 6,
        'random_state': 9999, 'verbosity': 0, 'tree_method': 'hist',
    },
    'optimized': {
        'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 8,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_lambda': 1.0, 'min_child_weight': 3,
        'random_state': 9999, 'verbosity': 0, 'tree_method': 'hist',
    },
}

print("\n" + "=" * 60)
print("XGBoost Classifier Results (512-bit Morgan fingerprints)")
print("=" * 60)

results = []
for cfg_name, params in configs.items():
    print(f"\n--- Config: {cfg_name} ---")
    seeds = [42, 123, 333, 9999]
    metrics_list = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        clf = xgb.XGBClassifier(**{**params, 'random_state': seed})
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        metrics_list['accuracy'].append(acc)
        metrics_list['f1'].append(f1)
        metrics_list['precision'].append(prec)
        metrics_list['recall'].append(rec)

    metrics_summary = {}
    for metric in metrics_list:
        vals = metrics_list[metric]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        metrics_summary[metric] = {'mean': mean_v, 'std': std_v}
        print(f"  {metric}: {mean_v:.4f} ± {std_v:.4f}")

    results.append({
        'config': cfg_name,
        'metrics': metrics_summary,
    })

# ── Compare with GNN ──
print("\n" + "=" * 60)
print("Comparison with GNN classifier (AdvancedGCN)")
print("=" * 60)
print(f"{'Model':<20} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 68)

gnn = {'Accuracy': 0.8421, 'F1': 0.8426, 'Precision': 0.8426, 'Recall': 0.8426}
print(f"{'GNN (AdvancedGCN)':<20} {gnn['Accuracy']:<12.4f} {gnn['F1']:<12.4f} {gnn['Precision']:<12.4f} {gnn['Recall']:<12.4f}")

for res in results:
    m = res['metrics']
    print(f"{'XGBoost ' + res['config']:<20} "
          f"{m['accuracy']['mean']:<12.4f} {m['f1']['mean']:<12.4f} "
          f"{m['precision']['mean']:<12.4f} {m['recall']['mean']:<12.4f}")

# ── Save results ──
results_json = os.path.join(os.path.dirname(__file__), "results", "xgb_classifier_results.json")
os.makedirs(os.path.dirname(results_json), exist_ok=True)
import json
with open(results_json, 'w') as f:
    json.dump({
        'gnn': {'accuracy': 0.8421, 'f1': 0.8426, 'precision': 0.8426, 'recall': 0.8426},
        'xgb_results': results,
        'seed_list': [42, 123, 333, 9999],
        'features': '512-bit Morgan fingerprints',
        'note': 'XGBoost uses same fingerprint input as GNN to test feature fairness hypothesis.',
    }, f, indent=2)
print(f"\nResults saved to {results_json}")
print("\nDone.")
