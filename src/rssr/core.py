"""Numerical conventions: floating CHW images, explicit range, no silent scaling."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def image_tensor(x):
    x = torch.as_tensor(x)
    if x.ndim != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("Expected nonempty floating CHW image")
    if not torch.isfinite(x).all():
        raise ValueError("Image contains nonfinite samples; apply explicit mask first")
    return x


def resize(x, scale, mode="bicubic"):
    x = image_tensor(x)
    if scale not in (2, 4) or mode not in ("nearest", "bilinear", "bicubic"):
        raise ValueError("Scale must be 2 or 4; unsupported interpolation")
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(x[None], scale_factor=scale, mode=mode, **kwargs)[0]


def degrade(x, scale=2, sigma=0.8, noise=0.0, seed=0):
    """Gaussian correlation (symmetric kernel), replicate edges, area sampling.

    Synthetic operator only. Noise is additive Gaussian in input units; no clipping.
    """
    x = image_tensor(x)
    if scale not in (2, 4) or any(v % scale for v in x.shape[-2:]):
        raise ValueError("Spatial dimensions must be divisible by scale 2 or 4")
    if not np.isfinite([sigma, noise]).all() or sigma < 0 or noise < 0:
        raise ValueError("Nonnegative finite sigma/noise required")
    if sigma:
        radius = int(np.ceil(3 * sigma))
        t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-t.square() / (2 * sigma**2)); k = k / k.sum()
        kernel = (k[:, None] * k[None, :]).expand(x.shape[0], 1, -1, -1)
        y = F.conv2d(F.pad(x[None], (radius,) * 4, mode="replicate"), kernel, groups=x.shape[0])
    else:
        y = x[None]
    y = F.avg_pool2d(y, scale)[0]
    generator = torch.Generator(device=x.device).manual_seed(seed)
    return y + noise * torch.randn(y.shape, generator=generator, device=x.device, dtype=x.dtype)


def metrics(pred, ref, data_range=1.0, border=0, mask=None):
    """MSE, RMSE, PSNR, valid-window SSIM, SAM radians, per-band RMSE.

    SSIM: uniform 7x7 population moments, K1=.01 K2=.03, average bands/windows.
    Zero-norm spectra are excluded from SAM (undefined), not assigned zero angle.
    """
    p, r = image_tensor(pred).double(), image_tensor(ref).double()
    if p.shape != r.shape or data_range <= 0 or not np.isfinite(data_range):
        raise ValueError("Matching shapes and positive finite data range required")
    if not isinstance(border, int) or border < 0 or 2 * border >= min(p.shape[-2:]):
        raise ValueError("Invalid border")
    m = torch.ones(p.shape[-2:], dtype=torch.bool, device=p.device) if mask is None else torch.as_tensor(mask, device=p.device).bool()
    if m.shape != p.shape[-2:]:
        raise ValueError("Mask must have shape HW")
    if border:
        p, r, m = p[:, border:-border, border:-border], r[:, border:-border, border:-border], m[border:-border, border:-border]
    if not m.any():
        raise ValueError("No valid pixels")
    error = (p-r).square()[:, m]
    mse = error.mean().item()
    norms = p.norm(dim=0) * r.norm(dim=0)
    valid = m & (norms > 1e-12)
    sam = torch.acos(((p*r).sum(0)[valid] / norms[valid]).clamp(-1, 1)).mean().item() if valid.any() else None
    ssim = None
    if min(p.shape[-2:]) >= 7:
        avg = lambda x: F.avg_pool2d(x[None], 7, 1)[0]
        a, b = avg(p), avg(r)
        va, vb = (avg(p*p)-a*a).clamp_min(0), (avg(r*r)-b*b).clamp_min(0)
        cov = avg(p*r)-a*b
        c1, c2 = (.01*data_range)**2, (.03*data_range)**2
        score = ((2*a*b+c1)*(2*cov+c2))/((a*a+b*b+c1)*(va+vb+c2))
        windows = F.avg_pool2d(m.double()[None, None], 7, 1)[0, 0] == 1
        if windows.any():
            ssim = score[:, windows].mean().item()
    return {"mse": mse, "rmse": mse**.5, "psnr": float("inf") if mse == 0 else 10*np.log10(data_range**2/mse), "ssim": ssim, "sam_rad": sam,
            **{f"rmse_band_{i}": v for i, v in enumerate(error.mean(1).sqrt().tolist())}}
