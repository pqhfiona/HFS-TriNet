#!/usr/bin/env python3
import json
import glob
import os

def main():
    base = "eval_speckle_results/var_0.050/run_0/20251227_172720/denoise_t_s_grid"
    rows = []
    for path in glob.glob(os.path.join(base, "*/summary.json")):
        try:
            with open(path) as f:
                arr = json.load(f)
        except Exception:
            continue
        if not arr:
            continue
        item = arr[0]
        item["label"] = os.path.basename(os.path.dirname(path))
        rows.append(item)
    rows_sorted = sorted(rows, key=lambda x: x.get('psnr_mean', 0), reverse=True)
    out = os.path.join(base, "combined_summary.json")
    with open(out, "w") as f:
        json.dump(rows_sorted, f, indent=2)
    print("WROTE", out)

if __name__ == "__main__":
    main()


