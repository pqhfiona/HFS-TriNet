import argparse
from pathlib import Path
import csv
import numpy as np
import cv2
import os

try:
    from skimage.metrics import structural_similarity as compare_ssim
except Exception:
    compare_ssim = None

def read_first_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None

def load_frames_from_dir(frame_dir, tmpl='img_{:05}.jpg', picks=(1,4,8)):
    frames = []
    for i in picks:
        p = Path(frame_dir) / tmpl.format(i)
        if not p.exists():
            files = sorted(Path(frame_dir).glob('*.jpg'))
            if not files:
                raise FileNotFoundError(f'No jpg files in {frame_dir}')
            p = files[min(i-1, len(files)-1)]
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f'Failed to read {p}')
        frames.append(img)
    return frames

def apply_speckle(frames, var):
    out = []
    for f in frames:
        arr = f.astype('float32') / 255.0
        noise = np.random.normal(0.0, np.sqrt(var), arr.shape).astype('float32')
        noisy = arr + arr * noise
        noisy = np.clip(noisy, 0.0, 1.0)
        out.append((noisy * 255.0).astype('uint8'))
    return out

def denoise_fastnlm(frames, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    out = []
    for f in frames:
        if f.ndim == 2:
            d = cv2.fastNlMeansDenoising(f, None, h, templateWindowSize, searchWindowSize)
        else:
            d = cv2.fastNlMeansDenoisingColored(f, None, h, hColor, templateWindowSize, searchWindowSize)
        out.append(d)
    return out

def psnr(a, b):
    return cv2.PSNR(a, b)

def ssim_gray(a, b):
    if compare_ssim is None:
        return float('nan')
    import numpy as _np
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    a_gray = a_gray.astype('float32')
    b_gray = b_gray.astype('float32')
    return compare_ssim(a_gray, b_gray, data_range=b_gray.max() - b_gray.min())

def save_grid(out_path, orig, noisy, denoised, label):
    rows = []
    for o,n,d in zip(orig, noisy, denoised):
        row = np.concatenate([o, n, d], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    cv2.putText(grid, label, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.imwrite(str(out_path), grid)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='predictions CSV path')
    parser.add_argument('--var', type=float, default=0.05)
    parser.add_argument('--outdir', default='eval_speckle_results/denoise_sweep')
    parser.add_argument('--picks', default='1,4,8')
    parser.add_argument('--hmin', type=int, default=None, help='minimum h for sweep (inclusive)')
    parser.add_argument('--hmax', type=int, default=None, help='maximum h for sweep (inclusive)')
    parser.add_argument('--step', type=int, default=1, help='step for h sweep')
    parser.add_argument('--t', type=int, default=None, help='templateWindowSize override')
    parser.add_argument('--s', type=int, default=None, help='searchWindowSize override')
    args = parser.parse_args()

    row = read_first_csv(args.csv)
    if row is None:
        raise RuntimeError('CSV is empty')
    frame_dir = row['patient_filename']
    picks = tuple(int(x) for x in args.picks.split(',') if x.strip())
    orig = load_frames_from_dir(frame_dir, picks=picks)
    noisy = apply_speckle(orig, args.var)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # parameter grid for fastNlMeans: (h, hColor, template, search)
    if args.hmin is not None and args.hmax is not None:
        t_default = args.t if args.t is not None else 7
        s_default = args.s if args.s is not None else 31
        grid = []
        for h in range(args.hmin, args.hmax + 1, args.step):
            grid.append((h, h, t_default, s_default))
    else:
        grid = [
            (3,3,3,21),
            (5,5,7,21),
            (8,8,7,31),
            (10,10,7,31),
            (15,15,7,41)
        ]

    summary = []
    for (h, hC, t, s) in grid:
        den = denoise_fastnlm(noisy, h=h, hColor=hC, templateWindowSize=t, searchWindowSize=s)
        label = f'fastnlm h={h} hc={hC} t={t} s={s}'
        out_path = outdir / f'denoise_h{h}_hc{hC}_t{t}_s{s}.png'
        save_grid(out_path, orig, noisy, den, label)
        # compute metrics per-frame then mean
        psnrs = [psnr(o,d) for o,d in zip(orig, den)]
        ssims = [ssim_gray(o,d) for o,d in zip(orig, den)]
        summary.append({
            'h':h, 'hColor':hC, 't':t, 's':s,
            'psnr_mean': float(np.mean(psnrs)),
            'psnr_std': float(np.std(psnrs)),
            'ssim_mean': float(np.nanmean(ssims)) if compare_ssim is not None else None,
            'ssim_std': float(np.nanstd(ssims)) if compare_ssim is not None else None,
            'out_image': str(out_path)
        })

    # write CSV summary
    import json
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved sweep results to', outdir)

if __name__ == '__main__':
    main()


