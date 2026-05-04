"""
Table 4 / Table S9: Computational Efficiency Benchmark

Compare XGBoost (Morgan 2048-bit + 12 descriptors) vs GNN
training time and inference time for varying dataset sizes.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.path.basename(project_root) == '实验':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
FP_DIM = 2048
SEED = 42
N_TRIALS = 3
N_VALUES = [100, 500, 1000, 1916]
GNN_EPOCHS = 100
GNN_PATIENCE = 20
BATCH_SIZE = 64
HIDDEN_DIM = 64
RESULTS_FILE = 'external_results/computational_efficiency.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DESC_FUNCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.RingCount,
    Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings,
    Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
    Descriptors.NumHeteroatoms,
]


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm = torch.nn.LayerNorm(hidden)
        self.predict = torch.nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.norm(x)
        g = global_mean_pool(x, batch)
        return self.predict(g).squeeze()


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
        f = [
            min(atom.GetAtomicNum() / 20.0, 1.0),
            float(atom.GetFormalCharge()),
            min(atom.GetNumImplicitHs() / 4.0, 1.0),
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
        ]
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


def compute_features(smiles_list):
    fps, descs = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(FP_DIM, dtype=np.float32))
            descs.append(np.zeros(len(DESC_FUNCS), dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        arr = np.zeros(FP_DIM, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        d = np.array([fn(mol) for fn in DESC_FUNCS], dtype=np.float32)
        descs.append(d)
    return np.concatenate([np.array(fps), np.array(descs)], axis=1)


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


if __name__ == '__main__':
    print("=" * 60)
    print("Computational Efficiency Benchmark")
    print(f"Device: {device}")
    print("=" * 60)

    df_h, pc, sc = load_data()

    # Precompute features
    print("Precomputing features...")
    X_fp = compute_features(df_h[sc].values)
    y_all = df_h[pc].values.astype(float)

    # Precompute graphs
    print("Precomputing graphs...")
    graph_list = []
    valid_idx = []
    for i, smi in enumerate(df_h[sc].values):
        g = mol_to_graph(smi)
        if g is not None:
            g.y = torch.tensor([y_all[i]], dtype=torch.float)
            graph_list.append(g)
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    X_fp = X_fp[valid_idx]
    y_all = y_all[valid_idx]
    print(f"有效图: {len(graph_list)}")

    in_dim = graph_list[0].x.shape[1]
    results = {}

    for n in N_VALUES:
        print(f"\n--- n={n} ---")
        n_actual = min(n, len(graph_list))
        xgb_times = []
        gnn_times = []
        xgb_infer = []
        gnn_infer = []
        xgb_r2s = []
        gnn_r2s = []

        for trial in range(N_TRIALS):
            seed = SEED + trial * 111
            rng = np.random.RandomState(seed)
            chosen = rng.choice(len(graph_list), n_actual, replace=False)
            tr_idx, te_idx = train_test_split(
                chosen, test_size=0.2, random_state=seed)

            # XGBoost timing
            t0 = time.time()
            xgb_m = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                random_state=seed, verbosity=0)
            xgb_m.fit(X_fp[tr_idx], y_all[tr_idx])
            t_train = time.time() - t0

            t0 = time.time()
            xp = xgb_m.predict(X_fp[te_idx])
            t_infer = time.time() - t0
            xgb_r2 = float(r2_score(y_all[te_idx], xp))

            xgb_times.append(t_train)
            xgb_infer.append(t_infer / len(te_idx))
            xgb_r2s.append(xgb_r2)

            # GNN timing
            train_data = [graph_list[i] for i in tr_idx]
            test_data = [graph_list[i] for i in te_idx]
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

            torch.manual_seed(seed)
            model = SimpleGCN(in_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            t0 = time.time()
            best_val = float('inf')
            patience = 0
            for epoch in range(GNN_EPOCHS):
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    loss = F.mse_loss(model(batch), batch.y)
                    loss.backward()
                    optimizer.step()
                # Quick val on 20% of train as early stopping
                model.eval()
                val_loss = 0
                count = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    val_loss += F.mse_loss(model(batch), batch.y).item() * batch.num_graphs
                    count += batch.num_graphs
                    break  # just first batch
                val_loss /= count
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= GNN_PATIENCE:
                        break
            t_train_gnn = time.time() - t0

            t0 = time.time()
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    preds.append(model(batch.to(device)).cpu())
            gp = torch.cat(preds).numpy()
            t_infer_gnn = time.time() - t0
            gnn_r2 = float(r2_score(y_all[te_idx], gp))

            gnn_times.append(t_train_gnn)
            gnn_infer.append(t_infer_gnn / len(te_idx))
            gnn_r2s.append(gnn_r2)

            print(f"  Trial {trial+1}: XGB={t_train:.2f}s/{t_infer/len(te_idx)*1000:.1f}ms R²={xgb_r2:.4f} | "
                  f"GNN={t_train_gnn:.2f}s/{t_infer_gnn/len(te_idx)*1000:.1f}ms R²={gnn_r2:.4f}")

            del model
            torch.cuda.empty_cache()

        results[str(n)] = {
            'n_samples': n_actual,
            'xgb_train_time_mean': float(np.mean(xgb_times)),
            'xgb_train_time_std': float(np.std(xgb_times)),
            'xgb_infer_time_per_sample_mean': float(np.mean(xgb_infer)),
            'xgb_infer_time_per_sample_std': float(np.std(xgb_infer)),
            'xgb_r2_mean': float(np.mean(xgb_r2s)),
            'xgb_r2_std': float(np.std(xgb_r2s)),
            'gnn_train_time_mean': float(np.mean(gnn_times)),
            'gnn_train_time_std': float(np.std(gnn_times)),
            'gnn_infer_time_per_sample_mean': float(np.mean(gnn_infer)),
            'gnn_infer_time_per_sample_std': float(np.std(gnn_infer)),
            'gnn_r2_mean': float(np.mean(gnn_r2s)),
            'gnn_r2_std': float(np.std(gnn_r2s)),
        }

    output = {
        'description': 'Computational efficiency benchmark: XGBoost vs GNN training/inference time',
        'dataset': 'OPV high-PCE',
        'device': str(device),
        'n_trials': N_TRIALS,
        'xgb_params': 'Morgan 2048bit FP + 12 RDKit descriptors, n_estimators=500, max_depth=6',
        'gnn_params': f'2-layer GCN, hidden={HIDDEN_DIM}, epochs={GNN_EPOCHS}, batch_size={BATCH_SIZE}',
        'results': results,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {RESULTS_FILE}")
