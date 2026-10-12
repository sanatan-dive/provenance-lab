"""Frequency-domain forensic features (vectorized over a batch of uint8 RGB images [N,H,W,3]).

Generators upsample from low-res latents (transposed conv / interpolation), which leaves
periodic peaks and an unnatural high-frequency fall-off in the Fourier spectrum.
"""
import numpy as np

N_RADIAL = 16


def _gray(images):
    x = images.astype(np.float32) / 255.0
    return x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  # [N,H,W]


def _power(gray):
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))
    return np.abs(f) ** 2


def log_spectrum(images):
    """Per-image standardized log-magnitude spectrum, [N,H,W] float32 (input to the CNN branch)."""
    s = np.log1p(np.sqrt(_power(_gray(images))))
    s = (s - s.mean(axis=(1, 2), keepdims=True)) / (s.std(axis=(1, 2), keepdims=True) + 1e-6)
    return s.astype(np.float32)


def spectrum_input(images, size=96):
    """Spectrum for the CNN branch. Large images use a full-resolution centre crop (keeps the
    high-frequency generator artefacts at a fraction of the memory); small ones are used whole."""
    h, w = images.shape[1:3]
    if h > size and w > size:
        t, l = (h - size) // 2, (w - size) // 2
        images = images[:, t:t + size, l:l + size]
    return log_spectrum(images)


def _median3(gray):
    p = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    h, w = gray.shape[1:]
    shifts = [p[:, i:i + h, j:j + w] for i in range(3) for j in range(3)]
    return np.median(np.stack(shifts), axis=0)


def handcrafted_features(images):
    """[N, 19]: 16-bin radial log-power profile, high-freq energy ratio, residual std + kurtosis."""
    gray = _gray(images)
    power = _power(gray)
    h, w = gray.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2)
    bins = np.minimum((r / r.max() * N_RADIAL).astype(int), N_RADIAL - 1)
    logp = np.log1p(power)
    radial = np.stack([logp[:, bins == b].mean(1) for b in range(N_RADIAL)], axis=1)
    total = power.sum(axis=(1, 2)) + 1e-12
    hf_ratio = power[:, r > min(h, w) / 4].sum(1) / total  # beyond half the Nyquist radius
    res = (gray - _median3(gray)).reshape(len(gray), -1)
    res_std = res.std(1)
    z = (res - res.mean(1, keepdims=True)) / (res_std[:, None] + 1e-8)
    res_kurt = (z ** 4).mean(1) - 3.0
    return np.concatenate([radial, hf_ratio[:, None], res_std[:, None], res_kurt[:, None]],
                          axis=1).astype(np.float32)
