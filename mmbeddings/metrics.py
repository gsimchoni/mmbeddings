import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

def upper_triangular_values(D):
    return D[np.triu_indices_from(D, k=1)]

def frobenius_distance(true_embed_list, pred_embed_list):
    frob_distances = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')
        frob_norm = np.linalg.norm(D_true - D_pred, ord='fro') / np.linalg.norm(D_true, ord='fro')
        frob_distances.append(frob_norm)
    return np.mean(frob_distances)

def spearman_rank_correlation(true_embed_list, pred_embed_list, sample_size=1000000):
    spearman_corrs = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')

        # Flatten the matrices
        D_true_flat = upper_triangular_values(D_true).flatten()
        D_pred_flat = upper_triangular_values(D_pred).flatten()

        # Randomly sample indices
        if sample_size < len(D_true_flat):
            indices = np.random.choice(len(D_true_flat), sample_size, replace=False)
            D_true_sampled = D_true_flat[indices]
            D_pred_sampled = D_pred_flat[indices]
        else:
            D_true_sampled = D_true_flat
            D_pred_sampled = D_pred_flat

        corr, _ = spearmanr(D_true_sampled, D_pred_sampled)
        spearman_corrs.append(corr if corr is not None else 0.0)
    return np.mean(spearman_corrs)

def normalized_rmse(true_embed_list, pred_embed_list):
    nrmse_values = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')
        diff = upper_triangular_values(D_true - D_pred)
        mse = np.mean(diff ** 2)
        norm_factor = np.max(D_true) - np.min(D_true)
        nrmse = np.sqrt(mse) / norm_factor if norm_factor > 0 else np.sqrt(mse)
        nrmse_values.append(nrmse)
    return np.mean(nrmse_values)

def calculate_embedding_metrics(true_embed_list, pred_embed_list):
    frob_distance = frobenius_distance(true_embed_list, pred_embed_list)
    spearman_corr = spearman_rank_correlation(true_embed_list, pred_embed_list)
    nrmse = normalized_rmse(true_embed_list, pred_embed_list)
    return frob_distance, spearman_corr, nrmse
