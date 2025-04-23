# File: metrics.py
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).astype(np.uint8)


def quantile_rmse(x: np.ndarray, y: np.ndarray, q: float) -> float:
    mask = y >= np.quantile(y, q)
    return np.sqrt(np.mean((x[mask] - y[mask])**2))


def calculate_metrics(truths: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mse = np.mean((truths - preds)**2)
    mae = np.mean(np.abs(truths - preds))
    rmse = np.sqrt(mse)
    corr, _ = pearsonr(truths.ravel(), preds.ravel())

    ssim_scores = []
    psnr_scores = []
    for t, p in zip(truths, preds):
        img_t = normalize_uint8(t)
        img_p = normalize_uint8(p)
        ssim_scores.append(ssim(img_t, img_p))
        psnr_scores.append(psnr(img_t, img_p))

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'CorrR': corr,
        'SSIM': np.mean(ssim_scores),
        'PSNR': np.mean(psnr_scores),
    }



