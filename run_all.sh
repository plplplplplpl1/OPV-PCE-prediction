#!/usr/bin/env bash
# ==============================================================
# 全量实验复现流水线 — 按论文从头运行全部实验
# ==============================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PLOG="$DIR/external_results/full_run_${TIMESTAMP}/pipeline.log"
mkdir -p "$(dirname "$PLOG")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PLOG"; }
header() { echo "" >> "$PLOG"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$PLOG"; echo "$1" >> "$PLOG"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$PLOG"; }

echo "======================================================" > "$PLOG"
echo "  OPV PCE 全量实验复现 — $TIMESTAMP" >> "$PLOG"
echo "======================================================" >> "$PLOG"

# 清理旧结果（保留备份和报告脚本）
log "清理旧结果文件..."
mkdir -p external_results/_backup
# 保留 _gen_report.py, _backup/, full_run_*/
for f in external_results/*.json; do
    [ -f "$f" ] && mv "$f" external_results/_backup/
done
log "旧结果已移至 _backup/"

# 恢复无runner的数据文件
cp external_results/_backup/fingerprint_sensitivity.json external_results/ 2>/dev/null || log "WARN: 无 fingerprint_sensitivity.json 备份"
cp external_results/_backup/learning_curve_data.json external_results/ 2>/dev/null || log "WARN: 无 learning_curve_data.json 备份"
log "恢复无runner的数据文件"

# ============================================================
# Phase 1: CPU 实验（并行运行）
# ============================================================
header "Phase 1: CPU 实验"
CPU_START=$(date +%s)

run_cpu() {
    local name="$1" script="$2"
    log "[CPU] 开始: $name"
    local start=$(date +%s)
    python3 "experiments/$script" >> "$PLOG" 2>&1 && {
        local elapsed=$(( $(date +%s) - start ))
        log "[CPU] ✅ $name 完成 (${elapsed}s)"
    } || {
        local elapsed=$(( $(date +%s) - start ))
        log "[CPU] ❌ $name 失败 (${elapsed}s)"
    }
}

# 并行启动CPU实验
run_cpu "Baseline Models" "run_baseline_models.py" &
PID1=$!
run_cpu "Feature Ablation" "run_feature_ablation.py" &
PID2=$!
run_cpu "XGBoost Classifier" "run_xgb_classifier.py" &
PID3=$!
run_cpu "FP-only Classification" "run_fp_only_classification.py" &
PID4=$!

# 等第一批完
wait $PID1 $PID2 $PID3 $PID4
log "[CPU] 第一批完成"

# 第二批CPU
run_cpu "Nonparametric Bootstrap" "run_nonparametric_bootstrap.py" &
PID5=$!
run_cpu "Screening Simulation" "run_screening_simulation.py" &
PID6=$!
run_cpu "Decision Framework" "run_decision_framework.py" &
PID7=$!
run_cpu "Crossover Model" "run_crossover_model.py" &
PID8=$!
run_cpu "Simple GCN OPV" "run_simple_gcn_opv.py" &
PID9=$!
run_cpu "Fingerprint Sensitivity" "run_fingerprint_sensitivity.py" &
PID10=$!
run_cpu "XGBoost Hyperopt OPV" "run_xgb_hyperopt_opv.py" &
PID11=$!
run_cpu "Computational Efficiency" "run_computational_efficiency.py" &
PID12=$!
run_cpu "Residual Analysis" "run_residual_analysis.py" &
PID13=$!
run_cpu "Seed 444 Experiment" "run_seed444_experiment.py" &
PID14=$!

wait $PID5 $PID6 $PID7 $PID8 $PID9 $PID10 $PID11 $PID12 $PID13 $PID14
CPU_ELAPSED=$(( $(date +%s) - CPU_START ))
log "[CPU] 全部CPU实验完成 (${CPU_ELAPSED}s)"

# ============================================================
# Phase 2: GPU 实验（串行，避免显存冲突）
# ============================================================
header "Phase 2: GPU 实验"
GPU_START=$(date +%s)

run_gpu() {
    local name="$1" script="$2"
    log "[GPU] 开始: $name"
    local start=$(date +%s)
    python3 "experiments/$script" >> "$PLOG" 2>&1 && {
        local elapsed=$(( $(date +%s) - start ))
        log "[GPU] ✅ $name 完成 (${elapsed}s)"
    } || {
        local elapsed=$(( $(date +%s) - start ))
        log "[GPU] ❌ $name 失败 (${elapsed}s)"
    }
}

run_gpu "GNN Capacity Scan" "run_gnn_capacity_scan.py"
run_gpu "Embedding Reverse Ablation" "run_embedding_reverse_ablation.py"
run_gpu "Embedding Evolution" "run_embedding_evolution.py"
run_gpu "Uncertainty Quantification" "run_uncertainty.py"
run_gpu "End-to-End Pipeline" "run_end_to_end_pipeline.py"
run_gpu "Pure GNN Ablation" "run_pure_gnn_ablation.py"
run_gpu "MoleculeNet Cross-Validation" "run_molenet_cross_validation.py"
run_gpu "MoleculeNet XGBoost Optuna" "run_molenet_xgb_optuna.py"
run_gpu "NREL OPV Validation" "run_nrel_opv_validation.py"
run_gpu "QM9 Full GNN" "run_qm9_full_gnn.py"
run_gpu "QM9 Large Only" "run_qm9_large_only.py"
run_gpu "QM9 Scale Experiment" "run_qm9_scale_experiment.py"
run_gpu "CEPDB GNN Multiseed" "run_cepdb_gnn_multiseed.py"
run_gpu "CEPDB GNN n100-500" "run_cepdb_gnn_n100_500.py"
run_gpu "SSL Pretrain" "run_ssl_pretrain.py"
run_gpu "Pretrain-Finetune" "run_pretrain_finetune.py"

GPU_ELAPSED=$(( $(date +%s) - GPU_START ))
log "[GPU] 全部GPU实验完成 (${GPU_ELAPSED}s)"

# ============================================================
# Phase 3: 超参数搜索（2小时）
# ============================================================
header "Phase 3: GNN 超参数搜索 (200 Optuna trials)"
HP_START=$(date +%s)
run_gpu "GNN Hyperparameter Search 200" "run_gnn_hparam_search_200.py"
HP_ELAPSED=$(( $(date +%s) - HP_START ))
log "[HP] 超参数搜索完成 (${HP_ELAPSED}s)"

# ============================================================
# 生成评估报告
# ============================================================
header "生成评估报告"
log "运行 _gen_report.py..."
python3 external_results/_gen_report.py 2>&1 | tee -a "$PLOG"
# 复制报告到本次运行目录
cp external_results/full_run_*/evaluation_report.md "external_results/full_run_${TIMESTAMP}/" 2>/dev/null || true

# ============================================================
# 汇总
# ============================================================
TOTAL_ELAPSED=$(( $(date +%s) - CPU_START ))
echo "" >> "$PLOG"
echo "======================================================" >> "$PLOG"
echo "  全量实验复现完成" >> "$PLOG"
echo "  CPU 实验: ${CPU_ELAPSED}s" >> "$PLOG"
echo "  GPU 实验: ${GPU_ELAPSED}s" >> "$PLOG"
echo "  超参数搜索: ${HP_ELAPSED}s" >> "$PLOG"
echo "  总计: ${TOTAL_ELAPSED}s" >> "$PLOG"
echo "======================================================" >> "$PLOG"

log "全量实验复现完成！日志: $PLOG"
log "运行以下命令查看报告:"
log "  cat external_results/full_run_${TIMESTAMP}/evaluation_report.md"
