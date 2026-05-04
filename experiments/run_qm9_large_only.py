"""
P2-4 QM9: Large n-values (20000, 50000, 100000)
================================================
Lightweight version with reduced model complexity for memory efficiency.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import json, os, time, warnings

os.environ['RDKIT_SILENCE'] = '1'
warnings.filterwarnings('ignore')

DATA_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
TARGET = 'gap'
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'external_results', 'qm9_scale_large.json')
TEST_SIZE = 10000
VAL_FRAC = 0.1
BATCH_SIZE = 32  # small batch for GPU memory efficiency
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ─── 1. Load ────────────────────────────────────────────────────────────────
print('Loading QM9 data...', flush=True)
df = pd.read_csv(DATA_URL)
df = df[df['smiles'].notna() & df[TARGET].notna()].reset_index(drop=True)
y_all = df[TARGET].values.astype(np.float32)
print(f'Loaded {len(df)} molecules, {TARGET}={y_all.mean():.4f}±{y_all.std():.4f}', flush=True)

# ─── 2. Fingerprints ────────────────────────────────────────────────────────
print('Computing Morgan fingerprints...', flush=True)
t0 = time.time()
fp_list = []
for i, smi in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros(2048, dtype=np.float32)
    fp_list.append(arr)
X_fp = np.stack(fp_list, dtype=np.float32)
print(f'Fingerprints: {X_fp.shape}, time={time.time()-t0:.1f}s', flush=True)

# ─── 3. Graphs (simple) ─────────────────────────────────────────────────────
ATOM_DIM = 15  # compact atom features

def make_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    atoms = mol.GetAtoms()
    xs = []
    for a in atoms:
        f = [
            min(a.GetAtomicNum() / 10.0, 1.0),
            float(a.GetFormalCharge()) / 2.0,
            min(a.GetNumImplicitHs() / 4.0, 1.0),
            min(a.GetTotalNumHs() / 8.0, 1.0),
            float(a.GetDegree()) / 4.0,
            1.0 if a.GetIsAromatic() else 0.0,
            1.0 if a.IsInRing() else 0.0,
        ]
        hyb = a.GetHybridization()
        f.extend([
            1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0,
            1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0,
            1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0,
        ])
        sym = a.GetSymbol()
        for s in ['C', 'N', 'O', 'F', 'S', 'Cl']:
            f.append(1.0 if sym == s else 0.0)
        xs.append(f)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    if not edges:
        return None
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(xs, dtype=torch.float), edge_index=ei)

print('Computing graphs...', flush=True)
t0 = time.time()
graphs = []
valid = []
for i, smi in enumerate(df['smiles']):
    g = make_graph(smi)
    if g is not None:
        g.y = torch.tensor([y_all[i]], dtype=torch.float)
        graphs.append(g)
        valid.append(i)
valid = np.array(valid)
X_fp = X_fp[valid]
y_all = y_all[valid]
n_total = len(graphs)
print(f'Valid: {n_total}, time={time.time()-t0:.1f}s', flush=True)
in_dim = graphs[0].x.shape[1]
print(f'Graph input dim: {in_dim}', flush=True)

# ─── 4. Fixed test split ────────────────────────────────────────────────────
rng42 = np.random.RandomState(42)
test_idx = set(rng42.choice(n_total, TEST_SIZE, replace=False))
test_i = [i for i in range(n_total) if i in test_idx]
train_i = [i for i in range(n_total) if i not in test_idx]
test_arr = np.array(test_i)
train_arr = np.array(train_i)

X_test = X_fp[test_arr]
y_test = y_all[test_arr]

print(f'Train pool: {len(train_arr)}, Test: {len(test_arr)}', flush=True)

# ─── 5. Simple GNN ──────────────────────────────────────────────────────────
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim, h=32):  # small model for CPU
        super().__init__()
        self.c1 = GCNConv(in_dim, h)
        self.reg = torch.nn.Linear(h, 1)

    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index))
        return self.reg(global_mean_pool(x, d.batch)).squeeze()

# ─── 6. Train function ──────────────────────────────────────────────────────
def train_gnn(tr_idx, va_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_data = [graphs[i] for i in tr_idx]
    va_data = [graphs[i] for i in va_idx]
    tr_loader = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=BATCH_SIZE)

    m = SimpleGCN(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    best = float('inf')
    best_sd = None
    patience = 0

    for epoch in range(150):
        m.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            F.mse_loss(m(b), b.y).backward()
            opt.step()

        m.eval()
        vl = 0
        with torch.no_grad():
            for b in va_loader:
                b = b.to(device)
                vl += F.mse_loss(m(b), b.y, reduction='sum').item()
        vl /= len(va_data)

        if epoch % 15 == 0 and epoch > 0:
            torch.cuda.empty_cache()

        if vl < best:
            best = vl
            best_sd = {k: v.clone() for k, v in m.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break

    m.load_state_dict(best_sd)
    return m

@torch.no_grad()
def eval_gnn(m):
    m.eval()
    loader = DataLoader([graphs[i] for i in test_arr], batch_size=BATCH_SIZE)
    preds = []
    for b in loader:
        preds.append(m(b))
    yp = torch.cat(preds).numpy()
    return {'r2': float(r2_score(y_test, yp)),
            'mae': float(mean_absolute_error(y_test, yp)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, yp)))}

# ─── 7. Run Experiment ──────────────────────────────────────────────────────
N_VALUES = [20000, 50000, 100000]
SEEDS = [42, 123, 456]
results = {}

for n in N_VALUES:
    print(f'\n=== n={n} ===', flush=True)
    n_results = {'xgb': [], 'gnn': [], 'xgb_r2': [], 'gnn_r2': []}

    for trial, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed + n)
        chosen = rng.choice(train_arr, n, replace=False)
        tr, va = train_test_split(chosen, test_size=VAL_FRAC, random_state=int(seed + n))

        # XGB
        xgb_m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
            random_state=seed, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_fp[tr], y_all[tr], eval_set=[(X_fp[va], y_all[va])], verbose=False)
        xp = xgb_m.predict(X_test)
        xs = {'r2': float(r2_score(y_test, xp)),
              'mae': float(mean_absolute_error(y_test, xp)),
              'rmse': float(np.sqrt(mean_squared_error(y_test, xp)))}

        # GNN
        try:
            print(f'  Trial {trial+1}: training GNN...', flush=True)
            t0 = time.time()
            gm = train_gnn(tr, va, seed)
            gs = eval_gnn(gm)
            del gm
            print(f'  Trial {trial+1}: done in {time.time()-t0:.0f}s', flush=True)
        except Exception as e:
            import traceback
            print(f'  Trial {trial+1} GNN FAILED: {e}', flush=True)
            traceback.print_exc()
            gs = {'r2': -999, 'mae': -999, 'rmse': -999}

        print(f'  n={n} Trial {trial+1}: XGB={xs["r2"]:.4f} GNN={gs["r2"]:.4f}', flush=True)
        n_results['xgb'].append(xs)
        n_results['gnn'].append(gs)
        n_results['xgb_r2'].append(xs['r2'])
        n_results['gnn_r2'].append(gs['r2'])

    xa, ga = np.array(n_results['xgb_r2']), np.array(n_results['gnn_r2'])
    results[str(n)] = {
        'xgb_r2_mean': float(xa.mean()), 'xgb_r2_std': float(xa.std()),
        'gnn_r2_mean': float(ga.mean()), 'gnn_r2_std': float(ga.std()),
        'delta_r2_mean': float((ga - xa).mean()), 'delta_r2_std': float((ga - xa).std()),
        'n_trials': len(SEEDS),
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  → n={n}: XGB={xa.mean():.4f} GNN={ga.mean():.4f} Δ={(ga-xa).mean():.4f}',
          flush=True)

print(f'\nDone! Results saved to {RESULTS_FILE}', flush=True)
