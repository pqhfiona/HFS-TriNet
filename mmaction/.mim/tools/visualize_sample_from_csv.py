import argparse
from pathlib import Path
import csv
import numpy as np
import cv2

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
            # try clamp to available files
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

def denoise_frames(frames):
    den = []
    for f in frames:
        if f.ndim == 2:
            d = cv2.fastNlMeansDenoising(f, None, 10, 7, 21)
        else:
            d = cv2.fastNlMeansDenoisingColored(f, None, 10, 10, 7, 21)
        den.append(d)
    return den

def denoise_frames_method(frames, method='fastnlm'):
    out = []
    for f in frames:
        if method == 'fastnlm':
            if f.ndim == 2:
                d = cv2.fastNlMeansDenoising(f, None, 10, 7, 21)
            else:
                d = cv2.fastNlMeansDenoisingColored(f, None, 10, 10, 7, 21)
        elif method == 'bilateral':
            # bilateralFilter only supports 3-channel or single channel
            if f.ndim == 2:
                d = cv2.bilateralFilter(f, d=9, sigmaColor=75, sigmaSpace=75)
            else:
                d = cv2.bilateralFilter(f, d=9, sigmaColor=75, sigmaSpace=75)
        elif method == 'median':
            k = 3
            if f.ndim == 2:
                d = cv2.medianBlur(f, k)
            else:
                # apply per-channel
                chans = cv2.split(f)
                chans = [cv2.medianBlur(c, k) for c in chans]
                d = cv2.merge(chans)
        elif method == 'gaussian':
            if f.ndim == 2:
                d = cv2.GaussianBlur(f, (5,5), 1.0)
            else:
                d = cv2.GaussianBlur(f, (5,5), 1.0)
        else:
            # fallback to original noisy
            d = f
        out.append(d)
    return out

def save_multi_method_grid(out_path, orig, noisy, methods, var, picks):
    """
    Save grid: rows per picked frame; columns: orig | noisy | denoised(method1) | denoised(method2) | ...
    """
    import numpy as _np
    rows = []
    for i in range(len(orig)):
        entries = []
        o = orig[i]
        n = noisy[i]
        entries.append(o)
        entries.append(n)
        for m in methods:
            den = denoise_frames_method([n], m)[0]
            entries.append(den)
        row = _np.concatenate(entries, axis=1)
        rows.append(row)
    grid = _np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), grid)

def save_grid(out_path, orig, noisy, denoised):
    rows = []
    for o,n,d in zip(orig, noisy, denoised):
        row = np.concatenate([o, n, d], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), grid)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='predictions CSV path')
    parser.add_argument('--var', type=float, default=0.05, help='speckle variance')
    parser.add_argument('--methods', type=str, default='', help='comma-separated denoise methods to apply (fastnlm,bilateral,median,gaussian)')
    parser.add_argument('--out', default='sample_vis.png', help='output image path')
    parser.add_argument('--tmpl', default='img_{:05}.jpg', help='frame filename template')
    parser.add_argument('--picks', default='1,4,8', help='comma-separated frame indices to sample')
    args = parser.parse_args()

    row = read_first_csv(args.csv)
    if row is None:
        raise RuntimeError('CSV is empty')
    frame_dir = row['patient_filename']
    picks = tuple(int(x) for x in args.picks.split(',') if x.strip())
    orig = load_frames_from_dir(frame_dir, tmpl=args.tmpl, picks=picks)
    noisy = apply_speckle(orig, args.var)
    # parse methods
    methods = [m.strip() for m in args.methods.split(',')] if getattr(args, 'methods', None) else []
    if methods and len(methods) > 0:
        # save grid with multiple denoising methods
        save_multi_method_grid(Path(args.out), orig, noisy, methods, args.var, picks)
        print(f'saved {args.out} (orig | noisy | {" | ".join(methods)})')
    else:
        denoised = denoise_frames(noisy)
        save_grid(Path(args.out), orig, noisy, denoised)
        print(f'saved {args.out} (orig | noisy | denoised)')

if __name__ == '__main__':
    main()


