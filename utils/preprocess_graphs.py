"""
图数据预处理脚本：将 data/data.csv 中的 SMILES 转换为 PyTorch Geometric 格式的图数据，
并保存到 data/processed/opv_graphs_class.pt，供 main_class.py 加载使用。
"""

import os
import pandas as pd
import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("错误：RDKit 未安装，无法继续。")
    exit(1)

DATA_CSV = 'data/data.csv'
OUTPUT_PT = 'data/processed/opv_graphs_class.pt'
PCE_THRESHOLD = 3.0


def smiles_to_graph(smiles):
    """与 main_class.py 中完全相同的 SMILES→图转换函数（30维节点特征）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            is_aromatic = int(atom.GetIsAromatic())
            is_in_ring = int(atom.IsInRing())

            try:
                hybridization = int(atom.GetHybridization())
                num_h = atom.GetTotalNumHs()
                valence = atom.GetTotalValence()
                is_in_ring3 = int(atom.IsInRingSize(3))
                is_in_ring4 = int(atom.IsInRingSize(4))
                is_in_ring5 = int(atom.IsInRingSize(5))
                is_in_ring6 = int(atom.IsInRingSize(6))
            except Exception:
                hybridization = 0
                num_h = 0
                valence = 0
                is_in_ring3 = is_in_ring4 = is_in_ring5 = is_in_ring6 = 0

            common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            atom_type_features = [int(atomic_num == x) for x in common_atoms]
            degree_features = [int(degree == x) for x in range(5)]
            hybrid_features = [int(hybridization == x) for x in range(1, 6)]

            features = [
                atomic_num / 100.0,
                degree / 6.0,
                formal_charge / 8.0,
                num_h / 4.0,
                valence / 8.0,
                is_aromatic,
                is_in_ring,
                is_in_ring3,
                is_in_ring4,
                is_in_ring5,
                is_in_ring6,
            ] + atom_type_features + degree_features + hybrid_features

            atom_features.append(features)

        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])

        if len(edge_indices) == 0:
            return None

        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


def main():
    print("读取数据 ...")
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    print(f"  总行数: {len(df)}")

    pce_column = df.columns[2]
    smiles_column = df.columns[-1]
    print(f"  PCE列: {pce_column}, SMILES列: {smiles_column}")

    # 清洗
    df[pce_column] = pd.to_numeric(df[pce_column], errors='coerce')
    df[smiles_column] = df[smiles_column].astype(str).str.strip()
    df = df.dropna(subset=[pce_column, smiles_column])
    df = df[df[smiles_column] != 'nan'].reset_index(drop=True)
    print(f"  清洗后有效行数: {len(df)}")

    # 创建分类标签
    df['PCE_class'] = (df[pce_column] > PCE_THRESHOLD).astype(int)
    print(f"  高PCE(>3%): {df['PCE_class'].sum()} 条")
    print(f"  低PCE(≤3%): {(df['PCE_class'] == 0).sum()} 条")

    print("\n生成图数据（可能需要几分钟）...")
    graphs = []
    labels = []
    smiles_index = {}  # SMILES → 图索引
    failed = 0

    for i, row in df.iterrows():
        smiles = row[smiles_column]
        graph = smiles_to_graph(smiles)
        if graph is not None:
            idx = len(graphs)
            graphs.append(graph)
            labels.append(int(row['PCE_class']))
            smiles_index[smiles] = idx
        else:
            failed += 1

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i+1}/{len(df)} 行，成功: {len(graphs)}, 失败: {failed}")

    print(f"\n完成！成功: {len(graphs)}, 失败: {failed}")

    # 验证特征维度
    if graphs:
        print(f"  节点特征维度: {graphs[0].x.shape[1]}")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    payload = {
        'graphs': graphs,
        'labels': labels,
        'smiles_index': smiles_index,
        'pce_threshold': PCE_THRESHOLD,
        'feature_dim': graphs[0].x.shape[1] if graphs else 0,
    }
    torch.save(payload, OUTPUT_PT)
    print(f"\n已保存至: {OUTPUT_PT}")
    print(f"  文件大小: {os.path.getsize(OUTPUT_PT) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
