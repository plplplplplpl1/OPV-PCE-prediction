"""
P1-2 supplement: Feature ablation for Table 4

Systematically vary XGBoost feature configurations:
1. Morgan 4096-bit only (no descriptors)
2. Morgan 4096-bit + 12 core descriptors
3. Morgan 4096-bit + 217 all descriptors
4. Morgan 512-bit only (no descriptors)
5. Morgan 512-bit + 12 core descriptors

Output: external_results/feature_ablation.json
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.path.basename(project_root) == '实验':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)
print(f"工作目录: {os.getcwd()}")

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
SEED = 123  # consistent with "reg_v1" in Table 3

# Optuna-optimized hyperparameters (from manuscript section 4.3)
HPARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.0117,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.595,
    'colsample_bytree': 0.626,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'verbosity': 0,
}

# 12 core physicochemical descriptors
CORE_DESC_FUNCS = [
    ('MolWt', Descriptors.MolWt),
    ('MolLogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA),
    ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('RingCount', Descriptors.RingCount),
    ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('NumAliphaticRings', Descriptors.NumAliphaticRings),
    ('FractionCSP3', Descriptors.FractionCSP3),
    ('HeavyAtomCount', Descriptors.HeavyAtomCount),
    ('NumHeteroatoms', Descriptors.NumHeteroatoms),
]

# All 217 RDKit descriptors
ALL_DESC_FUNCS = list(getattr(Descriptors, '_descList', []))
if not ALL_DESC_FUNCS:
    # fallback: use known 217 descriptor names
    try:
        from rdkit.Chem.Descriptors import descList
        ALL_DESC_FUNCS = list(descList)
    except:
        ALL_DESC_FUNCS = [(f'desc{i}', lambda m, i=i: 0.0) for i in range(217)]
N_ALL_DESC = len(ALL_DESC_FUNCS)
print(f"RDKit descriptor count: {N_ALL_DESC}")


def compute_fingerprint(smiles, nbits=4096, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_descriptors(smiles, mode='none'):
    """Compute descriptors. mode: 'none', 'core12', 'all217'"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mode == 'none':
        return np.array([], dtype=np.float32)
    elif mode == 'core12':
        vals = []
        for _, fn in CORE_DESC_FUNCS:
            try:
                v = fn(mol)
                if v is None or np.isnan(v) or np.isinf(v):
                    v = 0.0
            except:
                v = 0.0
            vals.append(float(v))
        return np.array(vals, dtype=np.float32)
    elif mode == 'all217':
        vals = []
        for _, fn in ALL_DESC_FUNCS:
            try:
                v = fn(mol)
                if v is None:
                    v = 0.0
                v = float(v)
                if np.isnan(v) or np.isinf(v):
                    v = 0.0
                if abs(v) > 1e8:
                    v = 0.0
            except:
                v = 0.0
            vals.append(v)
        return np.array(vals, dtype=np.float64)


def make_features(smiles_list, fp_bits, desc_mode):
    X_list = []
    for smi in smiles_list:
        fp = compute_fingerprint(smi, nbits=fp_bits)
        desc = compute_descriptors(smi, mode=desc_mode)
        if fp is None:
            fp = np.zeros(fp_bits, dtype=np.float32)
        if desc is None:
            if desc_mode == 'none':
                desc = np.array([], dtype=np.float32)
            elif desc_mode == 'core12':
                desc = np.zeros(12, dtype=np.float32)
            elif desc_mode == 'all217':
                desc = np.zeros(len(ALL_DESC_FUNCS), dtype=np.float32)
        X_list.append(np.concatenate([fp, desc]))
    return np.array(X_list)


def load_data():
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pc = df.columns[2]
    sc = df.columns[-1]
    df[pc] = pd.to_numeric(df[pc], errors='coerce')
    df[sc] = df[sc].astype(str).str.strip()
    df = df.dropna(subset=[pc, sc])
    df = df[df[sc] != 'nan'].reset_index(drop=True)
    df_h = df[df[pc] > PCE_THRESHOLD].copy().reset_index(drop=True)
    print(f"高PCE样本: {len(df_h)}")
    return df_h, pc, sc


def run_config(name, fp_bits, desc_mode):
    """Run one feature ablation configuration."""
    print(f"\n  [{name}] fp={fp_bits}, desc={desc_mode}")
    X_all = make_features(df_h[sc].values, fp_bits=fp_bits, desc_mode=desc_mode)
    y_all = df_h[pc].values.astype(float)
    print(f"    特征维度: {X_all.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED)
    print(f"    训练: {len(X_train)}, 测试: {len(X_test)}")

    model = xgb.XGBRegressor(**HPARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = float(r2_score(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    print(f"    R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    return {
        'name': name,
        'fp_bits': fp_bits,
        'desc_mode': desc_mode,
        'n_features': X_all.shape[1],
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Feature ablation for Table 4")
    print("=" * 60)
    print(f"XGBoost hparams: {HPARAMS}")
    print(f"Data: {DATA_CSV}")
    print(f"Seed: {SEED}")

    df_h, pc, sc = load_data()

    configs = [
        ('Morgan 4096-bit',        4096, 'none'),
        ('Morgan 4096+12desc',     4096, 'core12'),
        ('Morgan 4096+217desc',    4096, 'all217'),
        ('Morgan 512-bit',         512,  'none'),
        ('Morgan 512+12desc',      512,  'core12'),
    ]

    results = []
    for name, fp_bits, desc_mode in configs:
        r = run_config(name, fp_bits, desc_mode)
        results.append(r)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:25s}  R²={r['r2']:.4f}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}")

    output = {
        'method': 'XGBoost feature ablation (Table 4)',
        'model_params': {k: v for k, v in HPARAMS.items() if k != 'verbosity'},
        'data_split_seed': SEED,
        'results': results,
    }

    out_path = 'external_results/feature_ablation.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")
