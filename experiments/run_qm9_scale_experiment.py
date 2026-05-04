"""
P2-4: QM9 Large-Scale Phase Transition Validation
=================================================
Compares XGBoost (Morgan fingerprints) vs GNN across training set sizes
on the QM9 dataset (133,885 molecules, HOMO-LUMO gap target).

Goal: Verify that the phase transition (tree models → GNN with increasing data)
is universal, extending from experimental OPV PCE to quantum-chemical properties.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import json, os, time, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['RDKIT_SILENCE'] = '1'
warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
import os
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '论文材料', '07_数据', 'raw', 'qm9.csv')
DATA_URL = CACHE if os.path.exists(CACHE) else 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
TARGET = 'gap'
TEST_SIZE = 10000
VAL_FRAC = 0.1
N_VALUES = [100, 500, 1000, 5000, 20000, 50000, 100000]
N_TRIALS = 3
GNN_EPOCHS = 200
GNN_PATIENCE = 30
BATCH_SIZE = 128
LR = 0.001
HIDDEN_DIM = 128
N_LAYERS = 2  # 2 layers to reduce memory
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'external_results', 'qm9_scale_results.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── 1. Load Data ────────────────────────────────────────────────────────────
print('Loading QM9 data...')
df = pd.read_csv(DATA_URL)
if 'smiles' not in df.columns and 'SMILES' in df.columns:
    df = df.rename(columns={'SMILES': 'smiles'})
print(f'Loaded {len(df)} molecules')
df = df[df['smiles'].notna() & df[TARGET].notna()].reset_index(drop=True)
print(f'Valid molecules: {len(df)}')

y_all = df[TARGET].values.astype(np.float32)
print(f'{TARGET}: mean={y_all.mean():.4f}, std={y_all.std():.4f}, '
      f'range=[{y_all.min():.4f}, {y_all.max():.4f}]')

# ─── 2. Precompute Morgan Fingerprints ──────────────────────────────────────
print('Computing Morgan fingerprints...')
t0 = time.time()
fp_list = []
for i, smi in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros(2048, dtype=np.float32)
    fp_list.append(arr)
    if (i + 1) % 20000 == 0:
        print(f'  FP: {i+1}/{len(df)}')
X_fp = np.stack(fp_list)
print(f'Fingerprints: {X_fp.shape}, time={time.time()-t0:.1f}s')

# ─── 3. Precompute Molecular Graphs ─────────────────────────────────────────
print('Computing molecular graphs...')
t0 = time.time()

ATOM_TYPES = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'B', 'Si']
ATOM_TYPE_DIM = len(ATOM_TYPES)
MAX_DEG = 5
HYBRID_TYPES = [Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2]

def mol_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    atoms = mol.GetAtoms()
    feats = []
    for atom in atoms:
        f = []
        atype = [0.0] * ATOM_TYPE_DIM
        sym = atom.GetSymbol()
        if sym in ATOM_TYPES:
            atype[ATOM_TYPES.index(sym)] = 1.0
        else:
            atype[-1] = 1.0
        f.extend(atype)
        d = min(atom.GetDegree(), MAX_DEG)
        deg_f = [0.0] * (MAX_DEG + 1)
        deg_f[d] = 1.0
        f.extend(deg_f)
        hyb = atom.GetHybridization()
        hyb_f = [0.0] * (len(HYBRID_TYPES) + 1)
        if hyb in HYBRID_TYPES:
            hyb_f[HYBRID_TYPES.index(hyb)] = 1.0
        else:
            hyb_f[-1] = 1.0
        f.extend(hyb_f)
        f.extend([
            min(atom.GetAtomicNum() / 20.0, 1.0),
            float(atom.GetFormalCharge()),
            min(atom.GetNumImplicitHs() / 4.0, 1.0),
            min(atom.GetTotalNumHs() / 8.0, 1.0),
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
        ])
        feats.append(f)
    if not feats:
        return None
    x = torch.tensor(np.array(feats, dtype=np.float32))
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    if not edges:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

graph_list = []
valid_idx = []
for i, smi in enumerate(df['smiles']):
    g = mol_to_graph(smi)
    if g is not None:
        graph_list.append(g)
        valid_idx.append(i)
    if (i + 1) % 20000 == 0:
        print(f'  Graph: {i+1}/{len(df)}, valid={len(valid_idx)}')

valid_idx = np.array(valid_idx)
X_fp = X_fp[valid_idx]
y_all = y_all[valid_idx]
n_total = len(graph_list)
print(f'Valid graphs: {n_total}, time={time.time()-t0:.1f}s')

# ─── 4. Fixed Train/Val/Test Split ──────────────────────────────────────────
test_rng = np.random.RandomState(42)
test_idx_set = set(test_rng.choice(n_total, TEST_SIZE, replace=False))
test_li = [i for i in range(n_total) if i in test_idx_set]
train_val_li = [i for i in range(n_total) if i not in test_idx_set]
test_idx = np.array(test_li)
train_val_idx = np.array(train_val_li)
print(f'Train+val: {len(train_val_idx)}, Test: {len(test_idx)}')

X_fp_test = X_fp[test_idx]
y_test = y_all[test_idx]
graph_test = [graph_list[i] for i in test_idx]

# Pre-assign targets to all graphs
for i, g in enumerate(graph_list):
    g.y = torch.tensor([y_all[i]], dtype=torch.float)

def rmse(y, y_pred):
    return float(np.sqrt(mean_squared_error(y, y_pred)))

# ─── 5. Define GNN ──────────────────────────────────────────────────────────
class QM9_GCN(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, HIDDEN_DIM))
        for _ in range(N_LAYERS - 1):
            self.convs.append(GCNConv(HIDDEN_DIM, HIDDEN_DIM))
        self.norm = torch.nn.LayerNorm(HIDDEN_DIM)
        self.predict = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_DIM * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.norm(x)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.predict(g).squeeze()

in_dim = graph_list[0].x.shape[1]
print(f'GNN input dim: {in_dim}')

# ─── 6. Training Function ───────────────────────────────────────────────────
def train_gnn(train_idx, val_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()

    train_data = [graph_list[i] for i in train_idx]
    val_data = [graph_list[i] for i in val_idx]
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = QM9_GCN(in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val_loss = float('inf')
    best_dict = None
    patience = 0

    for epoch in range(GNN_EPOCHS):
        model.train()
        tl = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * batch.num_graphs
        tl /= len(train_data)

        model.eval()
        vl = 0
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                p = model(batch)
                vl += F.mse_loss(p, batch.y).item() * batch.num_graphs
                preds.append(p.cpu())
                trues.append(batch.y.cpu())
        vl /= len(val_data)
        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()
        vr2 = r2_score(trues, preds)
        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= GNN_PATIENCE:
                break

        # Memory cleanup every 20 epochs
        if epoch % 20 == 0 and epoch > 0:
            torch.cuda.empty_cache()

    model.load_state_dict(best_dict)
    model = model.to(device)
    return model

def eval_gnn(model):
    model.eval()
    loader = DataLoader(graph_test, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch.to(device)).cpu())
    y_pred = torch.cat(preds).numpy()
    return {
        'r2': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': rmse(y_test, y_pred),
    }

# ─── 7. Run Experiment ──────────────────────────────────────────────────────
results = {}
seeds = [42, 123, 456]

for trial, seed in enumerate(seeds):
    print(f'\n--- Trial {trial+1}/{len(seeds)} (seed={seed}) ---')
    rng = np.random.RandomState(seed)

    for n in N_VALUES:
        print(f'  n={n}: ', end='')
        chosen = rng.choice(train_val_idx, min(n, len(train_val_idx)), replace=False)
        tr_idx, va_idx = train_test_split(chosen, test_size=VAL_FRAC,
                                          random_state=int(seed + n))

        # XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.01, reg_lambda=0.01,
            early_stopping_rounds=50, random_state=seed, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_fp[tr_idx], y_all[tr_idx],
                  eval_set=[(X_fp[va_idx], y_all[va_idx])], verbose=False)
        xp = xgb_m.predict(X_fp_test)
        xs = {'r2': float(r2_score(y_test, xp)),
              'mae': float(mean_absolute_error(y_test, xp)),
              'rmse': rmse(y_test, xp)}

        # GNN
        try:
            gnn_m = train_gnn(tr_idx, va_idx, seed)
            gs = eval_gnn(gnn_m)
            del gnn_m
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f'GNN failed: {e}')
            print(traceback.format_exc())
            torch.cuda.empty_cache()
            gs = {'r2': -999, 'mae': -999, 'rmse': -999}

        print(f'XGB={xs["r2"]:.4f} GNN={gs["r2"]:.4f}')

        # Save per-trial results keyed by n
        n_key = str(n)
        if n_key not in results:
            results[n_key] = {'xgb_r2': [], 'gnn_r2': [], 'xgb': [], 'gnn': []}
        results[n_key]['xgb_r2'].append(xs['r2'])
        results[n_key]['gnn_r2'].append(gs['r2'])
        results[n_key]['xgb'].append(xs)
        results[n_key]['gnn'].append(gs)

        # Intermediate save
        summary = {}
        for nk in N_VALUES:
            sk = str(nk)
            if sk in results and len(results[sk]['xgb_r2']) > 0:
                xa = np.array(results[sk]['xgb_r2'])
                ga = np.array(results[sk]['gnn_r2'])
                d = ga - xa
                summary[sk] = {
                    'xgb_r2_mean': float(xa.mean()),
                    'xgb_r2_std': float(xa.std()),
                    'gnn_r2_mean': float(ga.mean()),
                    'gnn_r2_std': float(ga.std()),
                    'delta_r2_mean': float(d.mean()),
                    'delta_r2_std': float(d.std()),
                    'n_trials': len(xa),
                }
        with open(RESULTS_FILE, 'w') as f:
            json.dump(summary, f, indent=2)

# ─── 8. Final Summary ───────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('FINAL SUMMARY')
print(f'='*60)
for n_str in sorted(results.keys(), key=lambda x: int(x)):
    r = results[n_str]
    xa, ga = np.array(r['xgb_r2']), np.array(r['gnn_r2'])
    print(f'n={int(n_str):>6d}: XGB={xa.mean():.4f}±{xa.std():.4f}  '
          f'GNN={ga.mean():.4f}±{ga.std():.4f}  Δ={(ga-xa).mean():.4f}±{(ga-xa).std():.4f}')

print(f'\nSaved to: {RESULTS_FILE}')
