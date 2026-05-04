import pandas as pd
import numpy as np
import os
import json
import argparse

# 命令行参数
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--data', default=None, help='指定数据文件路径')
_parser.add_argument('--prefix', default=None, help='结果文件前缀（如 baseline/merged）')
_args, _ = _parser.parse_known_args()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习相关包
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch未安装，将跳过深度学习模型")

# 尝试导入图神经网络相关包
try:
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
    GNN_AVAILABLE = True and TORCH_AVAILABLE
except ImportError:
    GNN_AVAILABLE = False
    print("PyTorch Geometric未安装，将跳过图神经网络模型")

# 导入RDKit（如果可用）
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    # 抑制RDKit的警告信息
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit未安装，将使用简单的分子特征")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取数据...")
# 优先使用 --data 参数，其次 data_merged.csv，最后 data.csv
if _args.data:
    _data_path = _args.data
else:
    _data_path = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
_prefix = _args.prefix or ('merged' if 'merged' in _data_path else 'baseline')
print(f"数据文件: {_data_path}  |  结果前缀: {_prefix}")
data = pd.read_csv(_data_path, encoding='latin-1')
# 删除全为空的未命名列
data = data.loc[:, ~(data.columns.str.startswith('Unnamed') & data.isnull().all())]
print(f"数据形状: {data.shape}")
print(f"列名: {data.columns.tolist()}")
print("\n前5行数据:")
print(data.head())

# 检查数据
print("\n数据信息:")
print(data.info())
print("\n缺失值统计:")
print(data.isnull().sum())

# 假设最后一列是分子式，第三列是PCE值
# 根据实际数据调整列索引
if data.shape[1] >= 3:
    pce_column = data.columns[2]  # 第三列
    smiles_column = data.columns[-1]  # 最后一列
    
    print(f"\nPCE列: {pce_column}")
    print(f"分子式列: {smiles_column}")
    
    # 检查PCE值的分布
    print(f"\nPCE值统计:")
    print(data[pce_column].describe())
    
    # 创建分类标签 (0-2.9%为第一类，其余为第二类)
    data['PCE_class'] = (data[pce_column] > 3.0).astype(int)
    print(f"\n分类分布:")
    print(data['PCE_class'].value_counts())
    
    # 检查分子式数据
    print(f"\n分子式样例:")
    print(data[smiles_column].head(10))
    
    # 分子特征提取函数
    def extract_molecular_features(smiles):
        """从SMILES提取增强的分子描述符"""
        if RDKIT_AVAILABLE:
            try:
                # 数据清洗：移除可能的空白字符
                smiles = str(smiles).strip()
                if not smiles or smiles == 'nan':
                    return None
                
                # 尝试解析分子，使用sanitize=False避免严格的化学检查
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is None:
                    return None
                
                # 尝试进行sanitization，如果失败则跳过
                try:
                    Chem.SanitizeMol(mol)
                except:
                    # 如果sanitization失败，尝试使用基本的分子信息
                    try:
                        # 重新尝试不进行sanitization的解析
                        mol = Chem.MolFromSmiles(smiles, sanitize=False)
                        if mol is None:
                            return None
                    except:
                        return None
                
                # 提取增强的特征集，每个特征都单独try-catch
                features = {}
                
                # 基本分子性质
                try:
                    features['MolWt'] = Descriptors.MolWt(mol)
                except:
                    features['MolWt'] = len(smiles) * 10
                
                try:
                    features['LogP'] = Descriptors.MolLogP(mol)
                except:
                    features['LogP'] = 0.0
                
                try:
                    features['NumHDonors'] = Descriptors.NumHDonors(mol)
                except:
                    features['NumHDonors'] = smiles.count('O') + smiles.count('N')
                
                try:
                    features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
                except:
                    features['NumHAcceptors'] = smiles.count('O') + smiles.count('N')
                
                try:
                    features['TPSA'] = Descriptors.TPSA(mol)
                except:
                    features['TPSA'] = 0.0
                
                # 键和环特征
                try:
                    features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
                except:
                    features['NumRotatableBonds'] = smiles.count('-')
                
                try:
                    features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
                except:
                    features['NumAromaticRings'] = smiles.count('c') // 6
                
                try:
                    features['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
                except:
                    features['NumSaturatedRings'] = (smiles.count('1') + smiles.count('2')) // 2
                
                try:
                    features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
                except:
                    features['NumAliphaticRings'] = smiles.count('C') // 6
                
                try:
                    features['RingCount'] = Descriptors.RingCount(mol)
                except:
                    features['RingCount'] = smiles.count('1') + smiles.count('2') + smiles.count('3')
                
                try:
                    features['FractionCsp3'] = Descriptors.FractionCsp3(mol)
                except:
                    features['FractionCsp3'] = 0.5
                
                try:
                    features['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
                except:
                    features['NumHeteroatoms'] = smiles.count('N') + smiles.count('O') + smiles.count('S')
                
                try:
                    features['BertzCT'] = Descriptors.BertzCT(mol)
                except:
                    features['BertzCT'] = len(smiles)
                
                # 新增的高级分子描述符
                try:
                    features['MolMR'] = Descriptors.MolMR(mol)  # 分子折射率
                except:
                    features['MolMR'] = features['MolWt'] * 0.1
                
                try:
                    features['LabuteASA'] = Descriptors.LabuteASA(mol)  # 表面积
                except:
                    features['LabuteASA'] = features['MolWt'] * 0.5
                
                try:
                    features['BalabanJ'] = Descriptors.BalabanJ(mol)  # Balaban指数
                except:
                    features['BalabanJ'] = 1.0
                
                try:
                    features['Chi0v'] = Descriptors.Chi0v(mol)  # 连接性指数
                except:
                    features['Chi0v'] = len(smiles) * 0.1
                
                try:
                    features['Chi1v'] = Descriptors.Chi1v(mol)
                except:
                    features['Chi1v'] = len(smiles) * 0.05
                
                try:
                    features['Kappa1'] = Descriptors.Kappa1(mol)  # Kappa形状指数
                except:
                    features['Kappa1'] = len(smiles) * 0.2
                
                try:
                    features['Kappa2'] = Descriptors.Kappa2(mol)
                except:
                    features['Kappa2'] = len(smiles) * 0.1
                
                try:
                    features['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)  # Hall-Kier alpha
                except:
                    features['HallKierAlpha'] = 0.0
                
                try:
                    features['EState_VSA1'] = Descriptors.EState_VSA1(mol)  # E-state VSA描述符
                except:
                    features['EState_VSA1'] = 0.0
                
                try:
                    features['EState_VSA2'] = Descriptors.EState_VSA2(mol)
                except:
                    features['EState_VSA2'] = 0.0
                
                try:
                    features['VSA_EState1'] = Descriptors.VSA_EState1(mol)
                except:
                    features['VSA_EState1'] = 0.0
                
                try:
                    features['VSA_EState2'] = Descriptors.VSA_EState2(mol)
                except:
                    features['VSA_EState2'] = 0.0
                
                try:
                    features['SlogP_VSA1'] = Descriptors.SlogP_VSA1(mol)  # SlogP VSA描述符
                except:
                    features['SlogP_VSA1'] = 0.0
                
                try:
                    features['SlogP_VSA2'] = Descriptors.SlogP_VSA2(mol)
                except:
                    features['SlogP_VSA2'] = 0.0
                
                try:
                    features['SMR_VSA1'] = Descriptors.SMR_VSA1(mol)  # SMR VSA描述符
                except:
                    features['SMR_VSA1'] = 0.0
                
                try:
                    features['SMR_VSA2'] = Descriptors.SMR_VSA2(mol)
                except:
                    features['SMR_VSA2'] = 0.0
                
                try:
                    features['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)  # PEOE VSA描述符
                except:
                    features['PEOE_VSA1'] = 0.0
                
                try:
                    features['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
                except:
                    features['PEOE_VSA2'] = 0.0
                
                # 原子计数特征
                try:
                    features['NumCarbon'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
                except:
                    features['NumCarbon'] = smiles.count('C')
                
                try:
                    features['NumNitrogen'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
                except:
                    features['NumNitrogen'] = smiles.count('N')
                
                try:
                    features['NumOxygen'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
                except:
                    features['NumOxygen'] = smiles.count('O')
                
                try:
                    features['NumSulfur'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
                except:
                    features['NumSulfur'] = smiles.count('S')
                
                # 特征工程：组合特征
                features['MolWt_LogP_Ratio'] = features['MolWt'] / (abs(features['LogP']) + 1)
                features['TPSA_MolWt_Ratio'] = features['TPSA'] / (features['MolWt'] + 1)
                features['HeavyAtom_Count'] = features['NumCarbon'] + features['NumNitrogen'] + features['NumOxygen'] + features['NumSulfur']
                features['Ring_Density'] = features['RingCount'] / (features['HeavyAtom_Count'] + 1)
                features['Aromatic_Ratio'] = features['NumAromaticRings'] / (features['RingCount'] + 1)
                
                return features
                
            except Exception as e:
                # 如果RDKit完全失败，使用字符串特征
                return extract_string_features(smiles)
        else:
            return extract_string_features(smiles)
    
    def extract_string_features(smiles):
        """基于字符串的分子特征提取"""
        try:
            smiles = str(smiles).strip()
            if not smiles or smiles == 'nan':
                return None
                
            features = {
                'Length': len(smiles),
                'NumC': smiles.count('C'),
                'NumN': smiles.count('N'),
                'NumO': smiles.count('O'),
                'NumS': smiles.count('S'),
                'NumP': smiles.count('P'),
                'NumF': smiles.count('F'),
                'NumCl': smiles.count('Cl'),
                'NumBr': smiles.count('Br'),
                'NumI': smiles.count('I'),
                'NumRings': smiles.count('1') + smiles.count('2') + smiles.count('3'),
                'NumDoubleBonds': smiles.count('='),
                'NumTripleBonds': smiles.count('#'),
                'NumBranches': smiles.count('('),
                'NumAromaticC': smiles.count('c'),
                'NumAromaticN': smiles.count('n'),
                'NumAromaticO': smiles.count('o'),
                'NumAromaticS': smiles.count('s')
            }
            return features
        except:
            return None
    
    # 提取分子特征
    print("\n正在提取分子特征...")
    print(f"RDKit可用: {RDKIT_AVAILABLE}")
    molecular_features = []
    valid_indices = []
    
    for i, smiles in enumerate(data[smiles_column]):
        if i < 5:  # 打印前5个样本的调试信息
            print(f"处理第{i+1}个分子: {smiles[:50]}...")
        
        features = extract_molecular_features(smiles)
        if features is not None:
            molecular_features.append(features)
            valid_indices.append(i)
        elif i < 5:
            print(f"第{i+1}个分子特征提取失败")
    
    # 创建特征DataFrame
    if len(molecular_features) > 0:
        feature_df = pd.DataFrame(molecular_features)
        print(f"成功提取 {len(feature_df)} 个分子的特征")
        print(f"特征列: {feature_df.columns.tolist()}")
        print(f"特征统计:\n{feature_df.describe()}")
    else:
        print("错误：没有成功提取任何分子特征！")
        print("尝试处理第一个SMILES字符串...")
        first_smiles = data[smiles_column].iloc[0]
        print(f"第一个SMILES: {first_smiles}")
        
        # 手动测试特征提取
        test_features = extract_molecular_features(first_smiles)
        print(f"测试结果: {test_features}")
        
        # 如果还是失败，创建虚拟特征
        print("创建虚拟特征作为备用方案...")
        for i in range(min(100, len(data))):
            smiles = data[smiles_column].iloc[i]
            dummy_features = {
                'Length': len(str(smiles)),
                'NumC': str(smiles).count('C'),
                'NumN': str(smiles).count('N'),
                'NumO': str(smiles).count('O'),
                'NumS': str(smiles).count('S')
            }
            molecular_features.append(dummy_features)
            valid_indices.append(i)
        
        feature_df = pd.DataFrame(molecular_features)
        print(f"使用虚拟特征，成功处理 {len(feature_df)} 个分子")
    
    # 过滤有效数据
    valid_data = data.iloc[valid_indices].copy()
    valid_data = valid_data.reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)
    
    # 合并特征
    X = feature_df.values
    y = valid_data['PCE_class'].values
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 定义MLP模型
    if TORCH_AVAILABLE:
        class MLPClassifier(nn.Module):
            def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
                super(MLPClassifier, self).__init__()
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 2))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
    
    # 图神经网络模型
    if GNN_AVAILABLE:
        from torch_geometric.nn import GATConv, GraphConv, SAGEConv
        
        class AdvancedGCNClassifier(nn.Module):
            """高级图卷积网络分类器 - 集成多种GNN架构"""
            def __init__(self, input_dim, edge_dim=None, hidden_dim=160, num_layers=4, dropout=0.3, num_classes=2):
                super(AdvancedGCNClassifier, self).__init__()
                self.num_layers = num_layers
                self.dropout = dropout
                self.hidden_dim = hidden_dim
                
                # 多种GNN层的组合
                self.gcn_convs = nn.ModuleList()
                self.gat_convs = nn.ModuleList()
                self.sage_convs = nn.ModuleList()
                
                # GCN分支
                self.gcn_convs.append(GCNConv(input_dim, hidden_dim))
                for _ in range(num_layers - 1):
                    self.gcn_convs.append(GCNConv(hidden_dim, hidden_dim))
                
                # GAT分支
                self.gat_convs.append(GATConv(input_dim, hidden_dim // 4, heads=4, dropout=dropout))
                for _ in range(num_layers - 1):
                    self.gat_convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout))
                
                # GraphSAGE分支
                self.sage_convs.append(SAGEConv(input_dim, hidden_dim))
                for _ in range(num_layers - 1):
                    self.sage_convs.append(SAGEConv(hidden_dim, hidden_dim))
                
                # 批归一化层
                self.batch_norms = nn.ModuleList()
                for _ in range(num_layers):
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 3))  # 3个分支拼接
                
                # 注意力融合机制
                self.attention_weights = nn.Parameter(torch.ones(3) / 3)  # GCN, GAT, SAGE权重
                
                # 多尺度池化 - 使用包装类
                class GlobalPoolWrapper(nn.Module):
                    def __init__(self, pool_fn):
                        super().__init__()
                        self.pool_fn = pool_fn
                    
                    def forward(self, x, batch):
                        return self.pool_fn(x, batch)
                
                self.global_pools = nn.ModuleList([
                    GlobalPoolWrapper(global_mean_pool),
                    GlobalPoolWrapper(global_max_pool),
                    GlobalPoolWrapper(global_add_pool)
                ])
                
                # 高级分类器
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 3 * 3, hidden_dim * 2),  # 3个分支 * 3种池化
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout // 2),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
                
                # 残差连接的投影层
                self.residual_proj = nn.Linear(input_dim, hidden_dim * 3)
            
            def forward(self, x, edge_index, batch, edge_attr=None):
                # 保存输入用于残差连接
                residual = self.residual_proj(x)
                
                # 三个分支的前向传播
                gcn_x = x
                gat_x = x
                sage_x = x
                
                for i in range(self.num_layers):
                    # GCN分支
                    gcn_x = self.gcn_convs[i](gcn_x, edge_index)
                    gcn_x = F.relu(gcn_x)
                    gcn_x = F.dropout(gcn_x, p=self.dropout, training=self.training)
                    
                    # GAT分支
                    gat_x = self.gat_convs[i](gat_x, edge_index)
                    gat_x = F.relu(gat_x)
                    gat_x = F.dropout(gat_x, p=self.dropout, training=self.training)
                    
                    # SAGE分支
                    sage_x = self.sage_convs[i](sage_x, edge_index)
                    sage_x = F.relu(sage_x)
                    sage_x = F.dropout(sage_x, p=self.dropout, training=self.training)
                    
                    # 拼接三个分支
                    combined_x = torch.cat([gcn_x, gat_x, sage_x], dim=1)
                    
                    # 添加残差连接
                    if i == 0:
                        combined_x = combined_x + residual
                    
                    # 批归一化
                    combined_x = self.batch_norms[i](combined_x)
                    
                    # 更新各分支输入
                    gcn_x = combined_x[:, :self.hidden_dim]
                    gat_x = combined_x[:, self.hidden_dim:2*self.hidden_dim]
                    sage_x = combined_x[:, 2*self.hidden_dim:]
                
                # 多尺度全局池化
                pooled_features = []
                final_x = torch.cat([gcn_x, gat_x, sage_x], dim=1)
                
                for pool_wrapper in self.global_pools:
                    pooled = pool_wrapper(final_x, batch)
                    pooled_features.append(pooled)
                
                # 拼接所有池化特征
                x = torch.cat(pooled_features, dim=1)
                
                # 分类
                x = self.classifier(x)
                return x
        
        # 保持原有的简单GCN作为备选
        class GCNClassifier(nn.Module):
            """简化的图卷积网络分类器"""
            def __init__(self, input_dim, edge_dim=None, hidden_dim=128, num_layers=3, dropout=0.4, num_classes=2):
                super(GCNClassifier, self).__init__()
                self.num_layers = num_layers
                self.dropout = dropout
                self.hidden_dim = hidden_dim
                
                # GCN层
                self.convs = nn.ModuleList()
                self.convs.append(GCNConv(input_dim, hidden_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                
                # 批归一化层
                self.batch_norms = nn.ModuleList()
                for _ in range(num_layers):
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                
                # 简化的分类器
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
            
            def forward(self, x, edge_index, batch, edge_attr=None):
                # 图卷积层前向传播
                for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                    x = conv(x, edge_index)
                    x = bn(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                
                # 全局平均池化
                x = global_mean_pool(x, batch)
                
                # 分类
                x = self.classifier(x)
                return x
        

    
    # 从SMILES创建图数据
    def smiles_to_graph(smiles):
        """增强的SMILES转图数据函数，包含更丰富的特征"""
        if not (GNN_AVAILABLE and RDKIT_AVAILABLE):
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 增强的原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                # 基本特征
                atomic_num = atom.GetAtomicNum()
                degree = atom.GetDegree()
                formal_charge = atom.GetFormalCharge()
                is_aromatic = int(atom.GetIsAromatic())
                is_in_ring = int(atom.IsInRing())
                
                # 扩展特征
                try:
                    hybridization = int(atom.GetHybridization())
                    num_h = atom.GetTotalNumHs()
                    valence = atom.GetTotalValence()
                    is_in_ring3 = int(atom.IsInRingSize(3))
                    is_in_ring4 = int(atom.IsInRingSize(4))
                    is_in_ring5 = int(atom.IsInRingSize(5))
                    is_in_ring6 = int(atom.IsInRingSize(6))
                except:
                    hybridization = 0
                    num_h = 0
                    valence = 0
                    is_in_ring3 = is_in_ring4 = is_in_ring5 = is_in_ring6 = 0
                
                # 原子类型编码（扩展版）
                common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]  # H, C, N, O, F, P, S, Cl, Br
                atom_type_features = [int(atomic_num == x) for x in common_atoms]
                
                # 度数编码
                degree_features = [int(degree == x) for x in range(5)]  # 0-4度
                
                # 杂化编码
                hybrid_features = [int(hybridization == x) for x in range(1, 6)]  # SP-SP3D
                
                features = [
                    atomic_num / 100.0,  # 归一化原子序数
                    degree / 6.0,        # 归一化度数
                    formal_charge / 8.0,  # 归一化形式电荷
                    num_h / 4.0,         # 归一化氢原子数
                    valence / 8.0,       # 归一化价数
                    is_aromatic,
                    is_in_ring,
                    is_in_ring3,
                    is_in_ring4,
                    is_in_ring5,
                    is_in_ring6
                ] + atom_type_features + degree_features + hybrid_features
                
                atom_features.append(features)
            
            # 简化的边构建（无向图）
            edge_indices = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # 添加双向边（无向图）
                edge_indices.extend([[i, j], [j, i]])
            
            if len(edge_indices) == 0:
                return None
            
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)
        except Exception as e:
            return None
    
    # 创建图数据
    print("\n正在创建图数据...")
    graph_data = []
    graph_labels = []
    graph_indices = []

    # 优先从预计算缓存加载
    graph_cache = 'data/processed/opv_graphs_class.pt'
    if GNN_AVAILABLE and os.path.exists(graph_cache):
        print(f"加载预计算图数据: {graph_cache}")
        cached = torch.load(graph_cache, weights_only=False)
        graph_data = cached['graphs']
        graph_labels = cached['labels']
        graph_indices = list(range(len(graph_data)))
        print(f"从缓存加载 {len(graph_data)} 个图")
    else:
        for i, smiles in enumerate(valid_data[smiles_column]):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                graph_data.append(graph)
                graph_labels.append(valid_data.iloc[i]['PCE_class'])
                graph_indices.append(i)
    
    print(f"成功创建 {len(graph_data)} 个图")
    
    # 训练和评估函数
    def train_pytorch_model(model, X_train, y_train, X_test, y_test, epochs=100):
        """训练PyTorch模型"""
        if not TORCH_AVAILABLE:
            return None, None
            
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 训练
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        # 预测
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_probs = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
            test_preds = torch.argmax(test_outputs, dim=1).numpy()
        
        return test_preds, test_probs
    
    def augment_graph_data(graph_data, graph_labels, augment_ratio=0.3):
        """图数据增强函数"""
        augmented_graphs = []
        augmented_labels = []
        
        for graph, label in zip(graph_data, graph_labels):
            augmented_graphs.append(graph)
            augmented_labels.append(label)
            
            # 随机选择一部分图进行增强
            if np.random.random() < augment_ratio:
                # 节点特征噪声增强
                noise_graph = graph.clone()
                noise_scale = 0.05
                noise = torch.randn_like(noise_graph.x) * noise_scale
                noise_graph.x = noise_graph.x + noise
                augmented_graphs.append(noise_graph)
                augmented_labels.append(label)
                
                # 边dropout增强
                if np.random.random() < 0.5:
                    edge_dropout_graph = graph.clone()
                    num_edges = edge_dropout_graph.edge_index.size(1)
                    keep_edges = torch.randperm(num_edges)[:int(num_edges * 0.9)]
                    edge_dropout_graph.edge_index = edge_dropout_graph.edge_index[:, keep_edges]
                    augmented_graphs.append(edge_dropout_graph)
                    augmented_labels.append(label)
        
        return augmented_graphs, augmented_labels
    
    def train_gnn_model(graph_data, graph_labels, model_class, model_name, test_ratio=0.2, use_ensemble=False):
        """增强的GNN训练函数，支持数据增强和集成学习"""
        if not GNN_AVAILABLE:
            return None, None, None
        
        # 数据增强
        if model_name == 'AdvancedGCN':
            print("应用数据增强...")
            graph_data, graph_labels = augment_graph_data(graph_data, graph_labels, augment_ratio=0.4)
            print(f"增强后图数据数量: {len(graph_data)}")
            
        # 划分图数据
        n_graphs = len(graph_data)
        indices = np.random.permutation(n_graphs)
        split_idx = int(n_graphs * (1 - test_ratio))
        val_idx = int(n_graphs * (1 - test_ratio - 0.1))  # 添加验证集
        
        train_graphs = [graph_data[i] for i in indices[:val_idx]]
        val_graphs = [graph_data[i] for i in indices[val_idx:split_idx]]
        test_graphs = [graph_data[i] for i in indices[split_idx:]]
        train_labels = [graph_labels[i] for i in indices[:val_idx]]
        val_labels = [graph_labels[i] for i in indices[val_idx:split_idx]]
        test_labels = [graph_labels[i] for i in indices[split_idx:]]
        
        # 为图数据添加标签
        for i, graph in enumerate(train_graphs):
            graph.y = torch.tensor([train_labels[i]], dtype=torch.long)
        for i, graph in enumerate(val_graphs):
            graph.y = torch.tensor([val_labels[i]], dtype=torch.long)
        for i, graph in enumerate(test_graphs):
            graph.y = torch.tensor([test_labels[i]], dtype=torch.long)
        
        # 创建数据加载器 - 使用更大的批量以提高训练速度
        batch_size = 64  # 增大批量大小以提高训练效率
        train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = GeoDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        test_loader = GeoDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        
        # 模型 - 使用平衡的参数以防止过拟合
        input_dim = train_graphs[0].x.size(1)
        
        if model_name == 'AdvancedGCN':
            model = model_class(input_dim, hidden_dim=160, num_layers=4, dropout=0.3)
            # 高级模型使用更复杂的优化策略
            optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3, betas=(0.9, 0.999))
            
            # 组合学习率调度器
            scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=1e-6
            )
            scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.7, patience=8
            )
            
            # 焦点损失 + 标签平滑
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.label_smoothing = label_smoothing
                    self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                
                def forward(self, inputs, targets):
                    ce_loss = self.ce_loss(inputs, targets)
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss
            
            criterion = FocalLoss(alpha=1, gamma=2, label_smoothing=0.1)
            max_epochs = 100  # 更多训练轮数
            patience = 25  # 更大的耐心值
            
        elif model_name == 'GAT':
            model = model_class(input_dim, hidden_dim=64, num_layers=2, heads=4)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )
            scheduler2 = None
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            max_epochs = 60
            patience = 15
        else:
            # 为GCN使用平衡参数，防止过拟合
            model = model_class(input_dim, hidden_dim=128, num_layers=3, dropout=0.4)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )
            scheduler2 = None
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            max_epochs = 60
            patience = 15
        
        # 早停机制 - 基于验证准确率
        best_val_acc = 0.0
        patience_counter = 0
        print_interval = 5
        
        print(f"开始训练增强的{model_name}模型...")
        
        for epoch in range(max_epochs):
            # 训练阶段
            model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_train_loss += loss.item()
                
                # 计算训练准确率
                pred = out.argmax(dim=1)
                train_correct += (pred == batch.y).sum().item()
                train_total += batch.y.size(0)
            
            # 验证阶段
            model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    total_val_loss += loss.item()
                    
                    # 计算验证准确率
                    pred = out.argmax(dim=1)
                    val_correct += (pred == batch.y).sum().item()
                    val_total += batch.y.size(0)
            
            # 计算平均损失和准确率
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # 学习率调度
            if scheduler2 is not None:  # AdvancedGCN使用双调度器
                scheduler1.step()  # CosineAnnealingWarmRestarts
                scheduler2.step(val_acc)  # ReduceLROnPlateau
            else:
                scheduler1.step(val_acc)  # 其他模型使用单调度器
            
            # 早停检查 - 基于验证准确率
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % print_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f'{model_name} Epoch {epoch+1}/{max_epochs}:')
                print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}')
                print(f'  Val - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}')
                print(f'  LR: {current_lr:.6f}, Best Val Acc: {best_val_acc:.4f}')
            
            # 早停
            if patience_counter >= patience:
                print(f'{model_name} 早停于第 {epoch+1} 轮，最佳验证准确率: {best_val_acc:.4f}')
                break
        
        # 加载最佳模型
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        # 预测
        print("开始最终测试...")
        model.eval()
        test_preds = []
        test_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.softmax(out, dim=1)[:, 1]
                preds = torch.argmax(out, dim=1)
                test_probs.extend(probs.numpy())
                test_preds.extend(preds.numpy())
        
        # 计算最终测试准确率
        if len(test_preds) > 0:
            test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
            print(f"最终测试准确率: {test_acc:.4f}")
        
        return np.array(test_preds), np.array(test_probs), np.array(test_labels)
    
    # 模型字典
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    # 存储结果
    results = {}
    
    print("\n开始训练传统机器学习模型...")
    
    # 训练传统机器学习模型
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 添加调试信息
        print(f"测试集真实标签分布: {np.bincount(y_test)}")
        print(f"{name} 预测标签分布: {np.bincount(y_pred)}")
        
        results[name] = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0, average='weighted'),
            'recall': recall_score(y_test, y_pred, zero_division=0, average='weighted'),
            'f1': f1_score(y_test, y_pred, zero_division=0, average='weighted')
        }
        
        print(f"{name} - 准确率: {results[name]['accuracy']:.4f}")
        print(f"{name} - 精确率: {results[name]['precision']:.4f}")
        print(f"{name} - 召回率: {results[name]['recall']:.4f}")
        print(f"{name} - F1分数: {results[name]['f1']:.4f}")
        print("-" * 50)
    
    # 训练深度学习模型
    if TORCH_AVAILABLE:
        print("\n训练MLP模型...")
        mlp_model = MLPClassifier(X_train.shape[1])
        mlp_preds, mlp_probs = train_pytorch_model(mlp_model, X_train, y_train, X_test, y_test)
        
        if mlp_preds is not None:
            print(f"MLP 预测标签分布: {np.bincount(mlp_preds)}")
            
            results['MLP'] = {
                'predictions': mlp_preds,
                'probabilities': mlp_probs,
                'accuracy': accuracy_score(y_test, mlp_preds),
                'precision': precision_score(y_test, mlp_preds, zero_division=0, average='weighted'),
                'recall': recall_score(y_test, mlp_preds, zero_division=0, average='weighted'),
                'f1': f1_score(y_test, mlp_preds, zero_division=0, average='weighted')
            }
            
            print("-" * 50)
    
    # 训练图神经网络 - 高级GCN和标准GCN
    if GNN_AVAILABLE and len(graph_data) > 0:
        # 训练多个高级GCN模型进行集成
        print("\n训练集成高级图卷积网络(Ensemble AdvancedGCN)...")
        print(f"图数据特征维度: {graph_data[0].x.shape[1] if len(graph_data) > 0 else 'N/A'}")
        
        ensemble_preds = []
        ensemble_probs = []
        ensemble_test_labels = None
        
        # 训练3个不同配置的AdvancedGCN模型
        for i in range(3):
            print(f"\n训练集成模型 {i+1}/3...")
            # 设置不同的随机种子以获得多样性
            torch.manual_seed(42 + i * 10)
            np.random.seed(42 + i * 10)
            
            preds, probs, test_labels = train_gnn_model(graph_data, graph_labels, AdvancedGCNClassifier, f"AdvancedGCN_Ensemble_{i+1}")
            if preds is not None:
                ensemble_preds.append(probs)  # 使用概率进行集成
                if ensemble_test_labels is None:
                    ensemble_test_labels = test_labels
        
        # 集成预测结果
        if len(ensemble_preds) > 0:
            # 平均概率
            avg_probs = np.mean(ensemble_preds, axis=0)
            final_preds = (avg_probs > 0.5).astype(int)
            
            print(f"\n集成AdvancedGCN 测试集真实标签分布: {np.bincount(ensemble_test_labels)}")
            print(f"集成AdvancedGCN 预测标签分布: {np.bincount(final_preds)}")
            
            results['Ensemble_AdvancedGCN'] = {
                'predictions': final_preds,
                'probabilities': avg_probs,
                'test_labels': ensemble_test_labels,
                'accuracy': accuracy_score(ensemble_test_labels, final_preds),
                'precision': precision_score(ensemble_test_labels, final_preds, zero_division=0, average='weighted'),
                'recall': recall_score(ensemble_test_labels, final_preds, zero_division=0, average='weighted'),
                'f1': f1_score(ensemble_test_labels, final_preds, zero_division=0, average='weighted')
            }
            
            print(f"集成AdvancedGCN - 准确率: {results['Ensemble_AdvancedGCN']['accuracy']:.4f}")
            print(f"集成AdvancedGCN - 精确率: {results['Ensemble_AdvancedGCN']['precision']:.4f}")
            print(f"集成AdvancedGCN - 召回率: {results['Ensemble_AdvancedGCN']['recall']:.4f}")
            print(f"集成AdvancedGCN - F1分数: {results['Ensemble_AdvancedGCN']['f1']:.4f}")
            print("-" * 50)
        
        # 训练单个高级GCN模型作为对比
        print("\n训练单个高级图卷积网络(AdvancedGCN)...")
        torch.manual_seed(42)  # 重置随机种子
        np.random.seed(42)
        adv_gcn_preds, adv_gcn_probs, adv_gcn_test_labels = train_gnn_model(graph_data, graph_labels, AdvancedGCNClassifier, "AdvancedGCN")
        
        if adv_gcn_preds is not None:
            print(f"AdvancedGCN 测试集真实标签分布: {np.bincount(adv_gcn_test_labels)}")
            print(f"AdvancedGCN 预测标签分布: {np.bincount(adv_gcn_preds)}")
            
            results['AdvancedGCN'] = {
                'predictions': adv_gcn_preds,
                'probabilities': adv_gcn_probs,
                'test_labels': adv_gcn_test_labels,
                'accuracy': accuracy_score(adv_gcn_test_labels, adv_gcn_preds),
                'precision': precision_score(adv_gcn_test_labels, adv_gcn_preds, zero_division=0, average='weighted'),
                'recall': recall_score(adv_gcn_test_labels, adv_gcn_preds, zero_division=0, average='weighted'),
                'f1': f1_score(adv_gcn_test_labels, adv_gcn_preds, zero_division=0, average='weighted')
            }
            
            print("-" * 50)
        
        # 训练标准GCN模型作为对比
        print("\n训练标准图卷积网络(GCN)...")
        gcn_preds, gcn_probs, gcn_test_labels = train_gnn_model(graph_data, graph_labels, GCNClassifier, "GCN")
        
        if gcn_preds is not None:
            print(f"GCN 测试集真实标签分布: {np.bincount(gcn_test_labels)}")
            print(f"GCN 预测标签分布: {np.bincount(gcn_preds)}")
            
            results['GCN'] = {
                'predictions': gcn_preds,
                'probabilities': gcn_probs,
                'test_labels': gcn_test_labels,
                'accuracy': accuracy_score(gcn_test_labels, gcn_preds),
                'precision': precision_score(gcn_test_labels, gcn_preds, zero_division=0, average='weighted'),
                'recall': recall_score(gcn_test_labels, gcn_preds, zero_division=0, average='weighted'),
                'f1': f1_score(gcn_test_labels, gcn_preds, zero_division=0, average='weighted')
            }
            
            print("-" * 50)
        

    
    # 绘制ROC曲线
    plt.figure(figsize=(12, 8))
    
    for name in results.keys():
        if name in ['GCN', 'AdvancedGCN', 'Ensemble_AdvancedGCN'] and 'test_labels' in results[name]:
            fpr, tpr, _ = roc_curve(results[name]['test_labels'], results[name]['probabilities'])
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
            roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('各模型ROC曲线比较')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{_prefix}_roc_curves.png', dpi=300, bbox_inches='tight')
    
    
    # 绘制混淆矩阵
    n_models = len(results)
    # 动态计算子图布局
    if n_models <= 3:
        rows, cols = 1, n_models
        figsize = (5*n_models, 4)
    elif n_models <= 6:
        rows, cols = 2, 3
        figsize = (15, 8)
    else:
        rows, cols = 3, 3
        figsize = (15, 12)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else axes
    
    print("\n=== 各模型混淆矩阵 ===")
    
    for i, (name, result) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        # 获取真实标签和预测结果
        if name in ['GCN', 'AdvancedGCN', 'Ensemble_AdvancedGCN'] and 'test_labels' in result:
            true_labels = result['test_labels']
            pred_labels = result['predictions']
        else:
            true_labels = y_test
            pred_labels = result['predictions']
        
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建标注文本（显示数量和百分比）
        annot_text = np.empty_like(cm).astype(str)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                annot_text[row, col] = f'{cm[row, col]}\n({cm_percent[row, col]:.1f}%)'
        
        # 绘制热力图
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', ax=axes[i], 
                   cbar_kws={'label': '样本数量'}, square=True)
        axes[i].set_title(f'{name}\n混淆矩阵', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('预测标签', fontsize=10)
        axes[i].set_ylabel('真实标签', fontsize=10)
        
        # 设置标签
        axes[i].set_xticklabels(['低效率(≤3.0)', '高效率(>3.0)'], rotation=0)
        axes[i].set_yticklabels(['低效率(≤3.0)', '高效率(>3.0)'], rotation=0)
        
        # 打印混淆矩阵详细信息
        print(f"\n{name} 混淆矩阵:")
        print(f"真负例(TN): {cm[0,0]}, 假正例(FP): {cm[0,1]}")
        print(f"假负例(FN): {cm[1,0]}, 真正例(TP): {cm[1,1]}")
        
        # 计算特异性和敏感性
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"特异性(Specificity): {specificity:.4f}")
        print(f"敏感性(Sensitivity/Recall): {sensitivity:.4f}")
    
    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'results/{_prefix}_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵图表已保存为 'results/{_prefix}_confusion_matrices.png'")
    
    
    # 性能对比
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        model_names = list(results.keys())
        scores = [results[name][metric] for name in model_names]
        
        bars = axes[i].bar(model_names, scores, alpha=0.7)
        axes[i].set_title(f'{metric_name}比较')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'results/{_prefix}_performance_comparison.png', dpi=300, bbox_inches='tight')
    
    
    # 打印详细结果
    print("\n=== 模型性能总结 ===")
    print(f"{'模型':<15} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    print("-" * 55)
    
    for name, result in results.items():
        print(f"{name:<15} {result['accuracy']:<8.4f} {result['precision']:<8.4f} "
              f"{result['recall']:<8.4f} {result['f1']:<8.4f}")
    
    print("\n所有图表已保存为PNG文件。")

    # 导出结构化结果供对比脚本使用
    _json_results = {}
    for name, res in results.items():
        _json_results[name] = {
            'accuracy':  float(res['accuracy']),
            'precision': float(res['precision']),
            'recall':    float(res['recall']),
            'f1':        float(res['f1']),
        }
        # AUC（若存在概率预测）
        if res.get('probabilities') is not None:
            _tl = res.get('test_labels', y_test)
            _pr = np.array(res['probabilities'])
            if _pr.ndim > 1:
                _pr = _pr[:, 1]
            try:
                from sklearn.metrics import roc_auc_score
                _json_results[name]['auc'] = float(roc_auc_score(_tl, _pr))
            except Exception:
                pass
    os.makedirs('results', exist_ok=True)
    _json_path = f'results/{_prefix}_metrics.json'
    with open(_json_path, 'w', encoding='utf-8') as _jf:
        json.dump(_json_results, _jf, ensure_ascii=False, indent=2)
    print(f"结构化指标已保存至: {_json_path}")

    
else:
    print("数据列数不足，请检查数据格式")