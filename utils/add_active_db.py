"""
合并 Active_Database.csv 的受体分子到 data/data.csv
- 对重复的 Acceptor SMILES 取平均 PCE
- 过滤掉 SMILES 已存在于 data.csv 的行
- 生成新 Code（格式 "ACT001", "ACT002", ...）
- 备份原始 data.csv → data/data_backup.csv
"""

import pandas as pd
import shutil
import os

DATA_CSV = 'data/data.csv'
ACTIVE_CSV = 'data/Active_Database.csv'
BACKUP_CSV = 'data/data_backup.csv'

print("读取 data/data.csv ...")
df_orig = pd.read_csv(DATA_CSV, encoding='latin-1')
# 确保列名符合预期
print(f"  原始行数: {len(df_orig)}")
print(f"  列名: {df_orig.columns.tolist()}")

# 删除全为空的未命名列，确保列名干净
df_orig = df_orig.loc[:, ~(df_orig.columns.str.startswith('Unnamed') & df_orig.isnull().all())]

# 清理 SMILES 列（去除首尾空白）
# SMILES 是最后一列（去掉空列后）
smiles_col_orig = df_orig.columns[-1]  # 应为 'SMILES'
df_orig[smiles_col_orig] = df_orig[smiles_col_orig].astype(str).str.strip()
existing_smiles = set(df_orig[smiles_col_orig].dropna().unique())
existing_smiles.discard('nan')  # 移除可能的 'nan' 字符串
print(f"  原始唯一 SMILES 数量: {len(existing_smiles)}")

print("\n读取 data/Active_Database.csv ...")
df_active = pd.read_csv(ACTIVE_CSV, encoding='latin-1')
print(f"  Active_Database 行数: {len(df_active)}")
print(f"  列名: {df_active.columns.tolist()}")

# 提取关键列：Acceptor, Acceptor SMILES, PCE
acceptor_col = 'Acceptor'
acceptor_smiles_col = 'Acceptor SMILES'
pce_col = 'PCE'

df_active[acceptor_smiles_col] = df_active[acceptor_smiles_col].astype(str).str.strip()
df_active[pce_col] = pd.to_numeric(df_active[pce_col], errors='coerce')

# 过滤掉 PCE 为 NaN 或 SMILES 无效的行
df_active = df_active.dropna(subset=[acceptor_smiles_col, pce_col])
df_active = df_active[df_active[acceptor_smiles_col] != 'nan']
print(f"  清洗后有效行数: {len(df_active)}")

# 对重复 Acceptor SMILES 取平均 PCE
df_grouped = df_active.groupby(acceptor_smiles_col, as_index=False).agg(
    Acceptor=(acceptor_col, 'first'),
    PCE=(pce_col, 'mean')
)
print(f"  去重后唯一受体数量: {len(df_grouped)}")

# 过滤掉已存在于 data.csv 的 SMILES
df_new = df_grouped[~df_grouped[acceptor_smiles_col].isin(existing_smiles)].copy()
print(f"  需要新增的受体数量: {len(df_new)}")

if len(df_new) == 0:
    print("\n没有需要新增的受体，退出。")
    exit(0)

# 生成新 Code（格式 "ACT001", "ACT002", ...）
df_new = df_new.reset_index(drop=True)
df_new['Code'] = [f"ACT{i+1:03d}" for i in range(len(df_new))]

# 构建新行（匹配 data.csv 格式：Name, Code, PCE(%), SMILES）
# 原 data.csv 列：Name, Code, PCE(%), SMILES
# 注意：data.csv 最后一列可能有逗号（检查一下）
orig_columns = df_orig.columns.tolist()
pce_col_name = orig_columns[2]  # 第三列是 PCE

df_append = pd.DataFrame({
    orig_columns[0]: df_new['Acceptor'],       # Name
    orig_columns[1]: df_new['Code'],            # Code
    pce_col_name: df_new['PCE'].round(4),      # PCE(%)
    smiles_col_orig: df_new[acceptor_smiles_col],  # SMILES
})

# 如果原始 df_orig 有多余的列（如末尾空列），对齐
for col in orig_columns:
    if col not in df_append.columns:
        df_append[col] = ''

df_append = df_append[orig_columns]

# 备份原始 data.csv
print(f"\n备份 {DATA_CSV} → {BACKUP_CSV} ...")
shutil.copy2(DATA_CSV, BACKUP_CSV)
print(f"  备份完成")

# 追加新行到 data.csv（写入临时文件后覆盖）
df_merged = pd.concat([df_orig, df_append], ignore_index=True)
tmp_path = DATA_CSV + '.tmp'
df_merged.to_csv(tmp_path, index=False, encoding='latin-1')
# 用 copy 覆盖（比 rename 更兼容 Windows 文件锁）
try:
    shutil.copy2(tmp_path, DATA_CSV)
    os.remove(tmp_path)
except PermissionError:
    # 备选：写到 data_merged.csv，供手动替换
    fallback = 'data/data_merged.csv'
    shutil.copy2(tmp_path, fallback)
    os.remove(tmp_path)
    print(f"\n[警告] data.csv 被其他程序锁定，合并结果已写入: {fallback}")
    print(f"  请关闭占用 data.csv 的程序，然后手动将 {fallback} 重命名为 {DATA_CSV}")
    DATA_CSV = fallback

print(f"\n合并完成！")
print(f"  原始行数: {len(df_orig)}")
print(f"  新增行数: {len(df_append)}")
print(f"  合并后行数: {len(df_merged)}")
print(f"  PCE 分布:")
pce_vals = pd.to_numeric(df_merged[pce_col_name], errors='coerce')
print(pce_vals.describe())
print(f"\n  高PCE(>3%)样本数: {(pce_vals > 3.0).sum()}")
print(f"  低PCE(≤3%)样本数: {(pce_vals <= 3.0).sum()}")
