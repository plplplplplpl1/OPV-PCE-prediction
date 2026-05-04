# 数据目录说明

本目录用于存放OPV材料数据集。

## 数据格式

数据文件应为CSV格式,包含以下列:

| 列名 | 说明 | 示例 |
|------|------|------|
| Name | 材料名称 | "Polymer-1" |
| Code | 材料编码 | "P001" |
| PCE(%) | 功率转换效率(百分比) | 5.67 |
| SMILES | 分子SMILES字符串 | "CC1=CC=C(C=C1)..." |

## 数据文件

由于文件较大,原始数据不包含在仓库中。请按以下方式准备数据:

### 选项1: 使用自己的数据

将您的数据文件命名为 `data.csv` 并放置在此目录。

### 选项2: 下载示例数据

```bash
# 从发布页面下载
wget https://github.com/yourusername/OPV-GNN-Prediction/releases/download/v1.0.0/data.csv
```

### 选项3: 使用Active Database扩展

如果您有Active Database数据:

```bash
# 将Active_Database.csv放在此目录
# 运行合并脚本
python add_active_db.py
```

## 预处理数据

预处理后的图数据将保存在 `processed/` 子目录:

```bash
python preprocess_graphs_class.py
```

生成的文件:
- `processed/opv_graphs_class.pt` - 分类任务图数据
- `processed/opv_3d_graphs_high_pce.pt` - 3D高PCE图数据
- `processed/opv_3d_graphs_low_pce.pt` - 3D低PCE图数据

## 数据统计

### 原始数据集 (data.csv)
- 样本数: 1719
- PCE范围: 0.01% - 18.77%
- 高PCE (>3%): 1029 (60%)
- 低PCE (≤3%): 690 (40%)

### 合并数据集 (data_merged.csv)
- 样本数: 3018
- PCE范围: 0.000002% - 18.77%
- 高PCE (>3%): 1916 (63%)
- 低PCE (≤3%): 1102 (37%)
- 新增: 1299个独特受体SMILES

## 注意事项

1. 确保SMILES字符串格式正确
2. PCE值应为正数
3. 避免重复样本
4. 检查异常值和缺失值

## 数据隐私

如果您的数据包含敏感信息,请不要上传到公开仓库。
