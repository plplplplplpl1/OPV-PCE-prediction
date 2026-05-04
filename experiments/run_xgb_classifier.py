"""
XGBoost Classifier for PCE > 3% vs ≤ 3% Classification
========================================================
Purpose: Compare XGBoost classifier accuracy with GNN classifier (0.8421)
to address the question: "Can XGBoost match GNN on classification?"
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import os, json, warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
RESULTS_FILE = os.path.join(BASE_DIR, 'external_results', 'xgb_classifier_results.json')

PCE_THRESHOLD = 3.0
N_FOLDS = 5
FP_DIM = 2048

# ─── Load data ──────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
pce_col = [c for c in df.columns if 'pce' in c.lower()][0]
smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df = df.dropna(subset=[pce_col, smiles_col]).reset_index(drop=True)
y = (df[pce_col].values > PCE_THRESHOLD).astype(int)
print(f'Samples: {len(df)}, High-PCE: {y.sum()}, Low-PCE: {(y==0).sum()}')

# ─── Feature Engineering ────────────────────────────────────────────────────
print('Computing features...')

# Morgan fingerprints
fp_list = []
for smi in df[smiles_col]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        arr = np.zeros(FP_DIM, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros(FP_DIM, dtype=np.float32)
    fp_list.append(arr)
X_fp = np.stack(fp_list)

# 12 RDKit descriptors
desc_funcs = [
    ('MolWt', Descriptors.MolWt), ('MolLogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA), ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors), ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('RingCount', Descriptors.RingCount), ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('FractionCSP3', Descriptors.FractionCSP3), ('HeavyAtomCount', Descriptors.HeavyAtomCount),
    ('NHOHCount', Descriptors.NHOHCount), ('NOCount', Descriptors.NOCount),
]
desc_list = []
for smi in df[smiles_col]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        desc_list.append([f(mol) for _, f in desc_funcs])
    else:
        desc_list.append([0.0] * len(desc_funcs))
X_desc = np.array(desc_list, dtype=np.float32)

# ─── 3. Full descriptor set (217) ───────────────────────────────────────────
print('Computing all RDKit descriptors...')
all_desc_names = [d for d in Descriptors.descList]
def compute_all_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(len(all_desc_names), dtype=np.float32)
    try:
        return np.array([fn(mol) for _, fn in all_desc_names], dtype=np.float32)
    except:
        return np.zeros(len(all_desc_names), dtype=np.float32)
X_desc_full = np.stack([compute_all_descriptors(smi) for smi in df[smiles_col]])
# Remove NaN/Inf
X_desc_full = np.nan_to_num(X_desc_full, nan=0.0, posinf=0.0, neginf=0.0)

print(f'Features: FP={X_fp.shape[1]}, desc12={X_desc.shape[1]}, descFull={X_desc_full.shape[1]}')

# ─── 4. XGBoost Classifier ──────────────────────────────────────────────────
print('\n' + '='*55)
print('XGBoost Classifier (5-Fold CV)')
print('='*55)

feature_sets = {
    'FP_2048': X_fp,
    'FP_2048+Desc12': np.concatenate([X_fp, X_desc], axis=1),
    'FP_2048+DescFull': np.concatenate([X_fp, X_desc_full], axis=1),
}

results = {}
for feat_name, X in feature_sets.items():
    print(f'\n--- {feat_name} (dim={X.shape[1]}) ---')
    cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1,
            use_label_encoder=False, eval_metric='logloss')
        xgb_clf.fit(X[tr_idx], y[tr_idx])
        pred = xgb_clf.predict(X[te_idx])

        cv_scores['accuracy'].append(accuracy_score(y[te_idx], pred))
        cv_scores['precision'].append(precision_score(y[te_idx], pred, zero_division=0))
        cv_scores['recall'].append(recall_score(y[te_idx], pred, zero_division=0))
        cv_scores['f1'].append(f1_score(y[te_idx], pred, zero_division=0))

    for metric in cv_scores:
        cv_scores[metric] = {
            'mean': float(np.mean(cv_scores[metric])),
            'std': float(np.std(cv_scores[metric])),
        }
    results[feat_name] = cv_scores
    print(f'  Accuracy={cv_scores["accuracy"]["mean"]:.4f}±{cv_scores["accuracy"]["std"]:.4f}')
    print(f'  Precision={cv_scores["precision"]["mean"]:.4f}±{cv_scores["precision"]["std"]:.4f}')
    print(f'  Recall={cv_scores["recall"]["mean"]:.4f}±{cv_scores["recall"]["std"]:.4f}')
    print(f'  F1={cv_scores["f1"]["mean"]:.4f}±{cv_scores["f1"]["std"]:.4f}')

# ─── 5. GNN reference ──────────────────────────────────────────────────────
gnn_ref = {'accuracy': 0.8421, 'precision': 0.8426, 'recall': 0.8426, 'f1': 0.8426}
print(f'\nGNN reference: Acc={gnn_ref["accuracy"]:.4f}')

# ─── 6. Summary ────────────────────────────────────────────────────────────
print(f'\n{"="*55}')
print('Summary: XGBoost Classifier vs GNN Classifier')
print(f'{"="*55}')
print(f'{"Feature Set":<25} {"Accuracy":<15} {"F1":<15}')
print('-'*55)
for feat_name in feature_sets:
    r = results[feat_name]
    print(f'{feat_name:<25} {r["accuracy"]["mean"]:<8.4f}±{r["accuracy"]["std"]:<.4f}  {r["f1"]["mean"]:<8.4f}±{r["f1"]["std"]:<.4f}')
print(f'{"GNN (AdvancedGCN)":<25} {gnn_ref["accuracy"]:<15.4f} {gnn_ref["f1"]:<15.4f}')
print(f'{"RF (from Table 2)":<25} {"0.8123":<15} {"0.8125":<15}')
print(f'{"GB (from Table 2)":<25} {"0.8092":<15} {"0.8094":<15}')

# Check: does any XGBoost config match or exceed GNN?
best_feat = max(results, key=lambda k: results[k]['accuracy']['mean'])
best_xgb = results[best_feat]
print(f'\nBest XGBoost ({best_feat}): '
      f'Acc={best_xgb["accuracy"]["mean"]:.4f}')
if best_xgb['accuracy']['mean'] >= gnn_ref['accuracy']:
    print('→ XGBoost matches or exceeds GNN classifier!')
else:
    delta = gnn_ref['accuracy'] - best_xgb['accuracy']['mean']
    print(f'→ GNN classifier still leads by {delta:.4f} accuracy')

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
summary = {
    'xgb_results': results,
    'gnn_reference': gnn_ref,
    'rf_reference': {'accuracy': 0.8123, 'f1': 0.8125},
    'gb_reference': {'accuracy': 0.8092, 'f1': 0.8094},
}
with open(RESULTS_FILE, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nResults saved to {RESULTS_FILE}')
print('Done.')
