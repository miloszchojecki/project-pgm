import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List

def calculate_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def compute_fid(features1: np.ndarray, features2: np.ndarray) -> float:
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    return calculate_fid(mu1, sigma1, mu2, sigma2)

def compute_metrics(labels: List[int], preds: List[int]) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='binary'),
        "recall": recall_score(labels, preds, average='binary'),
        "f1": f1_score(labels, preds, average='binary')
    }
