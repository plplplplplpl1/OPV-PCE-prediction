import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise SystemExit(f"RDKit 未安装或不可用: {e}")

try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise SystemExit(f"catboost 未安装或不可用: {e}")


PCE_THRESHOLD = 3.0


def _get_data_path() -> str:
    return "data/data_merged.csv" if os.path.exists("data/data_merged.csv") else "data/data.csv"


def _infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    if df.shape[1] < 3:
        raise ValueError("数据列数不足，无法推断 PCE/SMILES 列")
    return df.columns[2], df.columns[-1]


def featurize(smiles: str, fp_bits: int = 2048, fp_radius: int = 2) -> np.ndarray | None:
    smiles = str(smiles).strip()
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_bits)
    fp_arr = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")

    desc = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.NOCount(mol),
        ],
        dtype=np.float32,
    )
    return np.concatenate([fp_arr.astype(np.float32), desc], axis=0)


def load_dataset(fp_bits: int, fp_radius: int) -> tuple[np.ndarray, np.ndarray]:
    data_csv = _get_data_path()
    df = pd.read_csv(data_csv, encoding="latin-1")
    pce_col, smiles_col = _infer_columns(df)

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)
    df = df[df[pce_col] > PCE_THRESHOLD].reset_index(drop=True)

    feats = []
    ys = []
    failed = 0
    for smi, pce in zip(df[smiles_col].tolist(), df[pce_col].tolist()):
        v = featurize(smi, fp_bits=fp_bits, fp_radius=fp_radius)
        if v is None:
            failed += 1
            continue
        feats.append(v)
        ys.append(float(pce))
    X = np.stack(feats, axis=0)
    y = np.array(ys, dtype=np.float32)
    print(f"数据文件: {data_csv}")
    print(f"高PCE样本 (PCE > {PCE_THRESHOLD}%): {len(df)} | 特征成功: {len(y)} | 失败: {failed}")
    print(f"特征维度: {X.shape[1]}")
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=3.0)
    args = parser.parse_args()

    X, y = load_dataset(fp_bits=args.fp_bits, fp_radius=args.fp_radius)

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    use_gpu = False
    task_type = "CPU"
    try:
        import torch

        if torch.cuda.is_available():
            use_gpu = True
            task_type = "GPU"
    except Exception:
        pass

    model = CatBoostRegressor(
        iterations=args.iters,
        learning_rate=args.lr,
        depth=args.depth,
        l2_leaf_reg=args.l2,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.seed,
        task_type=task_type,
        od_type="Iter",
        od_wait=500,
        verbose=200,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    print("\n测试集结果（PCE > 3% 子集）:")
    print(f"  MAE  = {mae:.4f} %")
    print(f"  RMSE = {rmse:.4f} %")
    print(f"  R2   = {r2:.4f}")

    result = {"model": "CatBoost", "seed": args.seed, "fp_bits": args.fp_bits,
              "MAE": mae, "RMSE": rmse, "R2": r2, "params": {"iters": args.iters,
              "depth": args.depth, "lr": args.lr, "l2": args.l2}}
    out_path = f"results/catboost_seed{args.seed}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()

