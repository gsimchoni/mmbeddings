import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

def frobenius_distance(true_embed_list, pred_embed_list):
    frob_distances = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')
        frob_norm = np.linalg.norm(D_true - D_pred, ord='fro') / np.linalg.norm(D_true, ord='fro')
        frob_distances.append(frob_norm)
    return np.mean(frob_distances)

def spearman_rank_correlation(true_embed_list, pred_embed_list):
    if any(X.shape[0] > 5000 for X in true_embed_list + pred_embed_list):
        return None
    spearman_corrs = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        D_true = cdist(X_true, X_true, metric='euclidean').flatten()
        D_pred = cdist(X_pred, X_pred, metric='euclidean').flatten()
        corr, _ = spearmanr(D_true, D_pred)
        spearman_corrs.append(corr if corr is not None else 0.0)
    return np.mean(spearman_corrs)

def normalized_rmse(true_embed_list, pred_embed_list):
    nrmse_values = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')
        mse = np.mean((D_true - D_pred) ** 2)
        norm_factor = np.max(D_true) - np.min(D_true)
        nrmse = np.sqrt(mse) / norm_factor if norm_factor > 0 else np.sqrt(mse)
        nrmse_values.append(nrmse)
    return np.mean(nrmse_values)

def calculate_embedding_metrics(true_embed_list, pred_embed_list):
    frob_distance = frobenius_distance(true_embed_list, pred_embed_list)
    spearman_corr = spearman_rank_correlation(true_embed_list, pred_embed_list)
    nrmse = normalized_rmse(true_embed_list, pred_embed_list)
    return frob_distance, spearman_corr, nrmse
