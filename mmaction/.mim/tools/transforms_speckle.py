import random
import numpy as np

class SpeckleNoise(object):
    """MM-style pipeline transform: 对每帧应用乘性speckle噪声."""
    def __init__(self, var=0.05, prob=1.0):
        self.var = float(var)
        self.prob = float(prob)

    def __call__(self, results):
        # results 通常包含 'imgs' 或 'img'：处理常见两种情况（list of frames 或 ndarray）
        if random.random() > self.prob:
            return results

        imgs = results.get('imgs', results.get('img', None))
        if imgs is None:
            return results

        def _apply_to_frame(frame):
            # frame: HWC uint8 or float
            arr = np.asarray(frame).astype(np.float32) / 255.0
            noise = np.random.normal(loc=0.0, scale=np.sqrt(self.var), size=arr.shape).astype(np.float32)
            noisy = arr + arr * noise
            noisy = np.clip(noisy, 0.0, 1.0)
            return (noisy * 255.0).astype(np.uint8)

        if isinstance(imgs, list):
            results['imgs'] = [ _apply_to_frame(f) for f in imgs ]
        else:
            # numpy array, 可能是 (T,H,W,C) 或 (H,W,C)
            arr = np.asarray(imgs)
            if arr.ndim == 4:  # T,H,W,C
                results['imgs'] = np.stack([ _apply_to_frame(arr[t]) for t in range(arr.shape[0]) ], axis=0)
            else:
                results['img'] = _apply_to_frame(arr)
        return results