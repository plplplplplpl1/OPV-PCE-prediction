#!/usr/bin/env python3
"""
=============================================================================
  OPV PCE 通宵可复刻性检验
  —— 全量实验重跑 + 论文对照，绝不覆盖现有结果 ——
=============================================================================

设计：
  - 备份现有 external_results/*.json → _backup/
  - 强制重新运行每一个实验
  - 捕获 stdout/stderr 提取 R² 等指标
  - 立即还原备份（绝不覆盖论文结果）
  - 逐项对比新鲜结果 vs 论文声明值
  - 生成综合报告

用法：
  python3 overnight_runner.py [--hours 10]

  --hours N  时间预算（默认 10 小时），超时自动终止并还原
"""

import os, sys, json, time, subprocess, traceback, datetime, shutil, re
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / 'external_results'
BACKUP_DIR = RESULTS_DIR / f'_overnight_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
MAIN_LOG = LOG_DIR / f'repro_{TIMESTAMP}.log'
REPORT_FILE = RESULTS_DIR / f'repro_report_{TIMESTAMP}.md'

T_START = time.time()
TIME_BUDGET = 10 * 3600  # default 10 hours

# ─── Paper reference values ──────────────────────────────────────────────
PAPER_VALUES = {
    'baseline_models': {
        'type': 'multi_seed_r2',
        'model': 'XGBoost',
        'expected_mean': 0.686,
        'expected_std': 0.026,
        'description': 'XGBoost multi-seed R² (Table 3)',
    },
    'feature_ablation': {
        'type': 'named_r2_list',
        'items': {
            'Morgan 4096-bit': 0.6942,
            'Morgan 4096+12desc': 0.7082,
            'Morgan 4096+217desc': 0.7181,
            'Morgan 512-bit': 0.7082,
            'Morgan 512+12desc': 0.7193,
        },
        'description': 'Feature ablation R² (Table 4)',
    },
    'fingerprint_sensitivity': {
        'type': 'named_r2_dict',
        'items': {
            'Morgan_512bit': 0.6843,
            'Morgan_2048bit': 0.6721,
            'Morgan_4096bit': 0.6635,
            'Morgan_4096bit_plus_8descriptors': 0.6961,
            'MACCS_keys_166bit': 0.6462,
            'RDKit_topological_fingerprint': 0.6637,
            'physicochemical_descriptors_only': 0.5679,
        },
        'description': 'Fingerprint sensitivity R² (Table 6)',
    },
    'qm9_full_gnn': {
        'type': 'nested_xgb_r2',
        'items': {'100': 0.618, '500': 0.740, '1000': 0.772, '5000': 0.851, '20000': 0.891, '50000': 0.904},
        'description': 'QM9 XGBoost R² (Table 13)',
    },
    'nrel_opv_validation': {
        'type': 'nested_xgb_r2',
        'items': {'100': 0.208, '500': 0.544, '1000': 0.639, '5000': 0.758, '20000': 0.814, '50000': 0.832},
        'description': 'NREL OPV XGBoost R² (Table 14)',
    },
    'molenet_cross_validation': {
        'type': 'dataset_xgb_r2',
        'items': {'ESOL': 0.676, 'FreeSolv': 0.731, 'Lipophilicity': 0.505},
        'description': 'MoleculeNet XGBoost R² (Table 12)',
    },
    'gnn_capacity_scan': {
        'type': 'capacity_r2',
        'items': {16: 0.570, 32: 0.574, 64: 0.584, 128: 0.603, 256: 0.581, 512: 0.586},
        'description': 'GNN capacity R² (Table 7)',
    },
    'gnn_hparam_search_200': {
        'type': 'single_value',
        'key': 'best_r2',
        'expected': 0.6515,
        'tolerance': 0.02,
        'description': 'GNN hyperparam search best R² (Table S4)',
    },
    'nonparametric_bootstrap': {
        'type': 'bootstrap_delta',
        'expected_lower': 0.052,
        'expected_upper': 0.134,
        'description': 'Bootstrap ΔR² 95% CI (Table S8)',
    },
    'xgb_classifier_results': {
        'type': 'gnn_accuracy',
        'expected': 0.8421,
        'description': 'GNN classifier accuracy (Table 2)',
    },
}

# ─── Experiments to run ──────────────────────────────────────────────────
EXPERIMENTS = [
    # (name, script, output_file, est_time_s, category, paper_key)
    ('XGBoost Baseline',       '实验/run_baseline_models.py',       'baseline_models.json',       300, 'cpu', 'baseline_models'),
    ('Feature Ablation',       '实验/run_feature_ablation.py',      'feature_ablation.json',      1800, 'cpu', 'feature_ablation'),
    ('XGBoost Classifier',     '实验/run_xgb_classifier.py',        'xgb_classifier_results.json', 600, 'cpu', 'xgb_classifier_results'),
    ('FP-only Classification', '实验/run_fp_only_classification.py','fp_only_classification.json', 120, 'cpu', None),
    ('Fingerprint Sensitivity','实验/run_fingerprint_sensitivity.py','fingerprint_sensitivity.json', 180, 'cpu', 'fingerprint_sensitivity'),
    ('XGBoost Hyperopt OPV',   '实验/run_xgb_hyperopt_opv.py',     'xgb_hyperopt_opv.json',      1800, 'cpu', None),
    ('Residual Analysis',      '实验/run_residual_analysis.py',     'residual_analysis.json',     120, 'cpu', None),
    ('Seed 444 Experiment',    '实验/run_seed444_experiment.py',    'seed444_results.json',       180, 'cpu', None),
    ('Nonparametric Bootstrap','实验/run_nonparametric_bootstrap.py','nonparametric_bootstrap.json', 360, 'cpu', 'nonparametric_bootstrap'),
    ('Screening Simulation',   '实验/run_screening_simulation.py',  'screening_simulation.json',  600, 'cpu', None),
    ('Crossover Model',        '实验/run_crossover_model.py',       'crossover_model.json',       60, 'cpu', None),
    ('Simple GCN OPV',         '实验/run_simple_gcn_opv.py',        'simple_gcn_opv.json',        240, 'cpu', None),
    ('GNN Capacity Scan',      '实验/run_gnn_capacity_scan.py',     'gnn_capacity_scan.json',     600, 'gpu_light', 'gnn_capacity_scan'),
    ('Embedding Evolution',    '实验/run_embedding_evolution.py',   'embedding_evolution_results.json', 180, 'gpu_light', None),
    ('Uncertainty Quant',      '实验/run_uncertainty.py',           'uncertainty_quantification.json', 360, 'gpu_light', None),
    ('Pure GNN Ablation',      '实验/run_pure_gnn_ablation.py',     'simple_gcn_opv.json',        300, 'gpu_light', None),
    ('CEPDB GNN n100-500',     '实验/run_cepdb_gnn_n100_500.py',    'cepdb_gnn_n100_500.json',    120, 'gpu_light', None),
    ('Embedding Reverse Ablation','实验/run_embedding_reverse_ablation.py','embedding_reverse_ablation.json', 900, 'gpu_medium', None),
    ('End-to-End Pipeline',    '实验/run_end_to_end_pipeline.py',    'end_to_end_pipeline.json',   600, 'gpu_medium', None),
    ('MoleculeNet CV',         '实验/run_molenet_cross_validation.py','molenet_cross_validation.json', 420, 'gpu_medium', 'molenet_cross_validation'),
    ('MoleculeNet XGB Optuna', '实验/run_molenet_xgb_optuna.py',    'molenet_xgb_optuna.json',    1200, 'gpu_medium', None),
    ('CEPDB GNN Multiseed',    '实验/run_cepdb_gnn_multiseed.py',   'cepdb_gnn_multiseed.json',   600, 'gpu_medium', None),
    ('SSL Pretrain',           '实验/run_ssl_pretrain.py',          'ssl_pretrain_results.json',  600, 'gpu_medium', None),
    ('Pretrain-Finetune',      '实验/run_pretrain_finetune.py',     'pretrain_finetune_results.json', 900, 'gpu_medium', None),
    ('Computational Efficiency','实验/run_computational_efficiency.py','computational_efficiency.json', 1800, 'gpu_heavy', None),
    ('NREL OPV Validation',    '实验/run_nrel_opv_validation.py',   'nrel_opv_validation.json',   3600, 'gpu_heavy', 'nrel_opv_validation'),
    ('QM9 Full GNN',           '实验/run_qm9_full_gnn.py',          'qm9_full_gnn.json',          3600, 'gpu_heavy', 'qm9_full_gnn'),
    ('QM9 Large Only',         '实验/run_qm9_large_only.py',        'qm9_scale_large.json',       3600, 'gpu_heavy', None),
    ('QM9 Scale Experiment',   '实验/run_qm9_scale_experiment.py',  'qm9_scale_results.json',     3600, 'gpu_heavy', None),
    ('GNN Hyperparam Search',  '实验/run_gnn_hparam_search_200.py', 'gnn_hparam_search_200.json', 10800, 'gpu_hparam', 'gnn_hparam_search_200'),
]


def log(msg):
    t = time.time() - T_START
    ts = f'[{t/3600:.1f}h]' if t > 3600 else f'[{t/60:.1f}min]'
    full = f'{ts} {msg}'
    print(full, flush=True)
    with open(MAIN_LOG, 'a') as f:
        f.write(full + '\n')


def backup_results():
    """Backup all JSON results before running anything."""
    log(f"备份现有结果 → {BACKUP_DIR}/")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in RESULTS_DIR.glob('*.json'):
        if '_backup' in str(f) or 'overnight' in str(f) or 'repro' in str(f):
            continue
        shutil.copy2(f, BACKUP_DIR / f.name)
        count += 1
    log(f"已备份 {count} 个文件")
    return count


def restore_results():
    """Restore all JSON results from backup."""
    log(f"还原备份 → {RESULTS_DIR}/")
    count = 0
    for f in BACKUP_DIR.glob('*.json'):
        shutil.copy2(f, RESULTS_DIR / f.name)
        count += 1
    log(f"已还原 {count} 个文件")


def extract_r2_from_output(output, script_name):
    """Extract R² values from script stdout for paper comparison."""
    r2_values = {}

    # Pattern: R²=0.xxxx or R2=0.xxxx or r2=0.xxxx
    r2_pattern = re.findall(r'R[\²2]\s*=\s*([0-9]+\.[0-9]+)', output)
    if r2_pattern:
        r2_values['all_r2'] = [float(x) for x in r2_pattern]

    # Pattern: R²=0.xxxx±0.xxxx (mean±std)
    r2ms = re.findall(r'R[\²2]\s*=\s*([0-9]+\.[0-9]+)±([0-9]+\.[0-9]+)', output)
    if r2ms:
        r2_values['r2_mean'] = float(r2ms[-1][0])
        r2_values['r2_std'] = float(r2ms[-1][1])

    return r2_values


def check_against_paper(exp_name, output_json_path, paper_key, output_text):
    """Compare experiment results against paper values."""
    if paper_key is None:
        return []

    ref = PAPER_VALUES.get(paper_key)
    if ref is None:
        return []

    checks = []
    try:
        data = json.load(open(output_json_path)) if output_json_path.exists() else {}
    except:
        data = {}

    r2_values = extract_r2_from_output(output_text, exp_name)

    if ref['type'] == 'multi_seed_r2':
        mean_r2 = r2_values.get('r2_mean')
        if mean_r2:
            ok = abs(mean_r2 - ref['expected_mean']) < 0.04
            checks.append((ok, f'{ref["description"]}: R²={mean_r2:.4f} (论文={ref["expected_mean"]}±{ref["expected_std"]})'))
        elif isinstance(data, dict) and 'results' in data:
            r2s = [r['r2'] for r in data['results'] if r.get('model') == ref['model']]
            if r2s:
                mn = np.mean(r2s)
                sd = np.std(r2s)
                ok = abs(mn - ref['expected_mean']) < 0.04
                checks.append((ok, f'{ref["description"]}: R²={mn:.4f}±{sd:.4f} (论文={ref["expected_mean"]}±{ref["expected_std"]})'))

    elif ref['type'] == 'named_r2_list':
        if isinstance(data, dict) and 'results' in data:
            for name, pr2 in ref['items'].items():
                found = False
                for r in data['results']:
                    if r.get('name') == name:
                        ok = abs(r['r2'] - pr2) < 0.001
                        checks.append((ok, f'{name}: R²={r["r2"]:.4f} (论文={pr2})'))
                        found = True
                        break
                if not found:
                    checks.append((False, f'{name}: 未找到结果'))

    elif ref['type'] == 'named_r2_dict':
        if isinstance(data, dict) and 'results' in data:
            results_dict = data['results']
            for name, pr2 in ref['items'].items():
                v = results_dict.get(name, {})
                if isinstance(v, dict) and 'r2' in v:
                    ok = abs(v['r2'] - pr2) < 0.002
                    checks.append((ok, f'{name}: R²={v["r2"]:.4f} (论文={pr2})'))

    elif ref['type'] == 'nested_xgb_r2':
        if isinstance(data, dict):
            for ns, pr2 in ref['items'].items():
                v = data.get(ns, {})
                xr = v.get('xgb_r2_mean', 0)
                ok = abs(xr - pr2) < 0.02
                checks.append((ok, f'n={ns}: XGB R²={xr:.4f} (论文={pr2})'))

    elif ref['type'] == 'dataset_xgb_r2':
        if isinstance(data, dict):
            for ds, pr2 in ref['items'].items():
                v = data.get(ds, {})
                sub = v.get('results', {}) if isinstance(v, dict) else {}
                # Find the largest n (full dataset) entry
                best_match = None
                for nk in sub:
                    if isinstance(sub[nk], dict) and 'xgb_r2_mean' in sub[nk]:
                        best_match = sub[nk]
                xa = best_match.get('xgb_r2_mean', 0) if best_match else 0
                ok = abs(xa - pr2) < 0.02
                checks.append((ok, f'{ds}: XGB R²={xa:.4f} (论文={pr2})'))

    elif ref['type'] == 'capacity_r2':
        if isinstance(data, dict):
            for hd, pr2 in ref['items'].items():
                v = data.get('gnn_results', {}).get(str(hd), {})
                mr2 = v.get('r2_mean', 0)
                ok = abs(mr2 - pr2) < 0.02
                checks.append((ok, f'h={hd}: GNN R²={mr2:.4f} (论文≈{pr2})'))

    elif ref['type'] == 'single_value':
        val = data.get(ref['key'], 0) if isinstance(data, dict) else 0
        tol = ref.get('tolerance', 0.01)
        ok = abs(val - ref['expected']) < tol
        checks.append((ok, f'{ref["description"]}: {val:.4f} (论文={ref["expected"]})'))

    elif ref['type'] == 'bootstrap_delta':
        if isinstance(data, dict):
            dr = data.get('delta_r2', {})
            ci = dr.get('ci_95', [0, 0])
            lower_ok = abs(ci[0] - ref['expected_lower']) < 0.005 if len(ci) > 0 else False
            upper_ok = abs(ci[1] - ref['expected_upper']) < 0.005 if len(ci) > 1 else False
            checks.append((lower_ok and upper_ok, f'{ref["description"]}: [{ci[0]:.4f},{ci[1]:.4f}] (论文=[{ref["expected_lower"]},{ref["expected_upper"]}])'))

    elif ref['type'] == 'gnn_accuracy':
        gr = data.get('gnn_reference', {}) if isinstance(data, dict) else {}
        acc = gr.get('accuracy', 0)
        ok = abs(acc - ref['expected']) < 0.001
        checks.append((ok, f'{ref["description"]}: {acc:.4f} (论文={ref["expected"]})'))

    return checks


def run_script(script_path, timeout=7200):
    """Run a script, capture output. Returns (success, stdout, stderr, returncode)."""
    abs_path = PROJECT_ROOT / script_path
    if not abs_path.exists():
        return False, '', f'Script not found: {script_path}', -1

    try:
        proc = subprocess.run(
            [sys.executable, str(abs_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        return proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        partial_out = e.stdout if hasattr(e, 'stdout') and e.stdout else ''
        partial_err = e.stderr if hasattr(e, 'stderr') and e.stderr else ''
        return False, partial_out, partial_err + '\nTIMEOUT', -1
    except Exception as e:
        return False, '', str(e), -1


def run_experiment_safe(exp):
    """Run one experiment with backup/restore protection."""
    name, script, out_file, est_time, cat, paper_key = exp

    log(f'\n{"="*60}')
    log(f'▶️  {name} ({cat}, 预计{est_time//60}分)')
    remaining = TIME_BUDGET - (time.time() - T_START)
    if remaining < est_time * 0.3:
        log(f'⚠️  时间不足，跳过')
        return {'name': name, 'success': False, 'reason': 'time_budget', 'checks': [], 'elapsed': 0}

    # Backup the specific output file before running
    orig_path = RESULTS_DIR / out_file
    backup_path = None
    if orig_path.exists():
        backup_path = BACKUP_DIR / out_file
        shutil.copy2(orig_path, backup_path)

    t0 = time.time()
    success, stdout, stderr, rc = run_script(str(script), timeout=min(int(est_time * 2.5), 14400))
    elapsed = time.time() - t0

    # Run paper checks on the (possibly overwritten) output
    output_path = RESULTS_DIR / out_file
    checks = check_against_paper(name, output_path, paper_key, stdout + stderr)

    # IMMEDIATELY restore the original from backup
    if backup_path and backup_path.exists():
        shutil.copy2(backup_path, orig_path)
        log(f'  🔄 已还原 {out_file}')

    # Log result
    passed = sum(1 for c in checks if c[0])
    total = len(checks)
    status = '✅' if success else '❌'

    log(f'  {status} {elapsed/60:.0f}分 (rc={rc}) | 论文对照: {passed}/{total} 通过')

    for ok, msg in checks:
        log(f'    {"✅" if ok else "❌"} {msg}')

    if stderr and len(stderr) > 200:
        log(f'  ⚠️  stderr: {stderr[:200]}...')

    return {
        'name': name,
        'success': success,
        'reason': 'ok' if success else f'rc={rc}',
        'checks': checks,
        'n_checks': total,
        'n_passed': passed,
        'elapsed': f'{elapsed:.0f}s',
        'stdout_tail': stdout[-300:] if stdout else '',
    }


def main():
    global TIME_BUDGET

    parser = argparse.ArgumentParser(description='OPV PCE Overnight Reproducibility Check')
    parser.add_argument('--hours', type=float, default=10, help='Time budget in hours')
    parser.add_argument('--quick', action='store_true', help='Quick: only CPU experiments + paper check')
    args = parser.parse_args()

    TIME_BUDGET = args.hours * 3600

    print(f'\n{"="*60}', flush=True)
    print(f'  OPV PCE 通宵可复刻性检验', flush=True)
    print(f'  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'  时间预算: {args.hours}小时', flush=True)
    print(f'  {"="*60}', flush=True)

    # Step 1: Backup all existing results
    n_backup = backup_results()
    if n_backup == 0:
        log('⚠️  没有找到现有结果文件备份')
    else:
        log(f'✅ 已备份 {n_backup} 个结果文件')

    all_results = []

    try:
        # Step 2: Run experiments
        cpu_exps = [e for e in EXPERIMENTS if e[4] == 'cpu']
        gpu_exps = [e for e in EXPERIMENTS if e[4] != 'cpu']

        if args.quick:
            log('\n📦 快速模式: 仅 CPU 实验')
            gpu_exps = []

        # CPU batch (parallel)
        log(f'\n📦 CPU 实验 ({len(cpu_exps)} 个)')
        from concurrent.futures import ThreadPoolExecutor, as_completed
        cpu_start = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(run_experiment_safe, exp): exp for exp in cpu_exps}
            for f in as_completed(futures):
                all_results.append(f.result())
        log(f'CPU 完成 ({time.time()-cpu_start:.0f}s)')

        # GPU batch (sequential)
        log(f'\n📦 GPU 实验 ({len(gpu_exps)} 个)')
        gpu_start = time.time()

        # Sort: light first, then medium, then heavy, then hparam
        priority = {'gpu_light': 0, 'gpu_medium': 1, 'gpu_heavy': 2, 'gpu_hparam': 3}
        gpu_exps.sort(key=lambda e: priority.get(e[4], 99))

        for exp in gpu_exps:
            remaining = TIME_BUDGET - (time.time() - T_START)
            if remaining < exp[3] * 0.3:
                log(f'⏭️  跳过 {exp[0]} (时间不足)')
                continue

            result = run_experiment_safe(exp)
            all_results.append(result)

            # Clean GPU memory
            import gc; gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except:
                pass

        # Step 3: Always restore everything
        restore_results()

        # Step 4: Generate report
        log('\n📊 生成复现性报告...')
        generate_report(all_results)

    except KeyboardInterrupt:
        log('\n⚠️ 被中断，正在还原...')
        restore_results()
        log('✅ 已还原')
        sys.exit(1)
    except Exception as e:
        log(f'\n❌ 错误: {e}')
        traceback.print_exc()
        log('正在还原...')
        restore_results()
        log('✅ 已还原')
        raise

    # Final message
    total_paper_checks = sum(r.get('n_checks', 0) for r in all_results)
    total_passed = sum(r.get('n_passed', 0) for r in all_results)
    log(f'\n{"="*60}')
    log(f'  通宵可复刻性检验完成!')
    log(f'  总用时: {(time.time()-T_START)/3600:.1f}h')
    log(f'  论文对照: {total_passed}/{total_paper_checks} 通过')
    log(f'  报告: {REPORT_FILE}')
    log(f'  {"="*60}')


def generate_report(results):
    """Generate comprehensive reproducibility report."""
    total_checks = sum(r.get('n_checks', 0) for r in results)
    total_passed = sum(r.get('n_passed', 0) for r in results)
    total_experiments = len(results)
    success_count = sum(1 for r in results if r['success'])
    failed_count = sum(1 for r in results if not r['success'] and r.get('reason') != 'time_budget')

    lines = [
        f'# OPV PCE 通宵可复刻性检验报告',
        f'',
        f'**时间**: {TIMESTAMP}',
        f'**总用时**: {(time.time()-T_START)/3600:.1f}h',
        f'**实验**: {total_experiments} | ✅ {success_count} 成功 | ❌ {failed_count} 失败',
        f'**论文对照**: ✅ {total_passed}/{total_checks} 项匹配',
        f'',
        f'---',
        f'## 逐项结果',
        f'',
        f'| 实验 | 状态 | 用时 | 论文对照 |',
        f'|------|------|------|---------|',
    ]

    for r in results:
        status = '✅' if r['success'] else '❌'
        if r.get('reason') == 'time_budget':
            status = '⏭️'
        pc = f'{r.get("n_passed", 0)}/{r.get("n_checks", 0)}' if r.get('n_checks', 0) > 0 else '—'
        lines.append(f'| {r["name"]} | {status} | {r.get("elapsed","?")} | {pc} |')

    lines.extend(['', '## 论文对照详情', ''])
    for r in results:
        if not r.get('checks'):
            continue
        all_ok = all(c[0] for c in r['checks'])
        lines.append(f'### {r["name"]} — {"✅ ALL PASS" if all_ok else "❌ HAS FAILURES"}')
        for ok, msg in r['checks']:
            lines.append(f'- {"✅" if ok else "❌"} {msg}')

    if failed_count > 0:
        lines.extend(['', '## 失败实验', ''])
        for r in results:
            if not r['success'] and r.get('reason') != 'time_budget':
                lines.append(f'- **{r["name"]}**: {r.get("reason","?")}')
                if r.get('stdout_tail'):
                    lines.append(f'  ```\n  {r["stdout_tail"]}\n  ```')

    lines.extend(['', '---', f'*报告由通宵自动化复现系统生成于 {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*', ''])

    report = '\n'.join(lines)
    with open(REPORT_FILE, 'w') as f:
        f.write(report)
    print(f'\n报告已保存: {REPORT_FILE}', flush=True)
    print(f'论文对照: {total_passed}/{total_checks} 通过', flush=True)


if __name__ == '__main__':
    main()
