import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

RESULTS_ROOT = Path('./eval_speckle_results')
OUT_PNG = RESULTS_ROOT / 'accuracy_vs_var.png'
OUT_CSV = RESULTS_ROOT / 'accuracy_vs_var.csv'

def load_all_results(path: Path):
    f = path / 'all_results_summary.json'
    if not f.exists():
        raise FileNotFoundError(f'Expected results file not found: {f}')
    with open(f, 'r') as fh:
        return json.load(fh)

def extract_metric_for_var(var_entry, metric_key='accuracy'):
    """从 var_entry（dict with per_repeat & aggregate）提取每次 repeat 的 metric 列表和聚合值"""
    per_repeat = var_entry.get('per_repeat', [])
    vals = []
    for r in per_repeat:
        if isinstance(r, dict) and metric_key in r:
            vals.append(float(r[metric_key]))
    # fallback: try any numeric key
    if not vals and per_repeat:
        for r in per_repeat:
            if isinstance(r, dict):
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                        break
    agg = var_entry.get('aggregate', {})
    mean = agg.get(metric_key, (np.mean(vals) if vals else float('nan')))
    std = (np.std(vals) if vals else float('nan'))
    return vals, float(mean), float(std)

def find_best_metric_key(all_results):
    # try 'accuracy' first, then common names
    common = ['accuracy', 'top1', 'top-1', 'top_1']
    sample = next(iter(all_results.values()))
    agg = sample.get('aggregate', {})
    for k in common:
        if k in agg:
            return k
    # fallback to any numeric key in aggregate
    for k, v in agg.items():
        if isinstance(v, (int, float)):
            return k
    return None

def main():
    all_results = load_all_results(RESULTS_ROOT)
    metric_key = find_best_metric_key(all_results)
    if metric_key is None:
        raise RuntimeError('No numeric metric key found in results aggregate')

    vars_list = []
    means = []
    stds = []
    for var_str, entry in sorted(all_results.items(), key=lambda x: float(x[0])):
        # keys are strings like "0.01"
        var_val = float(var_str)
        _, mean, std = extract_metric_for_var(entry, metric_key=metric_key)
        vars_list.append(var_val)
        means.append(mean)
        stds.append(std)

    # save CSV
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['noise_var', f'{metric_key}_mean', f'{metric_key}_std'])
        for v, m, s in zip(vars_list, means, stds):
            writer.writerow([v, m, s])

    # plot
    plt.figure(figsize=(6, 4))
    plt.errorbar(vars_list, means, yerr=stds, marker='o', capsize=4)
    plt.xlabel('Speckle variance')
    plt.ylabel(f'{metric_key} (mean ± std)')
    plt.title(f'{metric_key} vs Speckle variance')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=200)
    print(f'Plot saved to {OUT_PNG}, CSV saved to {OUT_CSV}')

if __name__ == '__main__':
    main()


