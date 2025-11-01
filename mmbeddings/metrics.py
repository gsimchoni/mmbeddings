import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from mmbeddings.utils import adjusted_auc

def upper_triangular_values(D):
    return D[np.triu_indices_from(D, k=1)]

def frobenius_distance(true_embed_list, pred_embed_list, sample_size = 10000):
    frob_distances = []
    
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        q = X_true.shape[0]
        
        if q > sample_size:
            indices = np.random.choice(q, sample_size, replace=False)
            X_true = X_true[indices]
            X_pred = X_pred[indices]
        
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')
        frob_norm = np.linalg.norm(D_true - D_pred, ord='fro') / np.linalg.norm(D_true, ord='fro')
        frob_distances.append(frob_norm)

    return np.mean(frob_distances)

def spearman_rank_correlation(true_embed_list, pred_embed_list, sample_size=10000, sample_size_corr=1000000):
    spearman_corrs = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        q = X_true.shape[0]
        
        if q > sample_size:
            indices = np.random.choice(q, sample_size, replace=False)
            X_true = X_true[indices]
            X_pred = X_pred[indices]
        
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')

        # Flatten the matrices
        D_true_flat = upper_triangular_values(D_true).flatten()
        D_pred_flat = upper_triangular_values(D_pred).flatten()

        # Randomly sample indices
        if sample_size_corr < len(D_true_flat):
            indices = np.random.choice(len(D_true_flat), sample_size, replace=False)
            D_true_sampled = D_true_flat[indices]
            D_pred_sampled = D_pred_flat[indices]
        else:
            D_true_sampled = D_true_flat
            D_pred_sampled = D_pred_flat

        corr, _ = spearmanr(D_true_sampled, D_pred_sampled)
        spearman_corrs.append(corr if corr is not None else 0.0)
    return np.mean(spearman_corrs)

def normalized_rmse(true_embed_list, pred_embed_list, sample_size=10000):
    nrmse_values = []
    for X_true, X_pred in zip(true_embed_list, pred_embed_list):
        q = X_true.shape[0]
        
        if q > sample_size:
            indices = np.random.choice(q, sample_size, replace=False)
            X_true = X_true[indices]
            X_pred = X_pred[indices]
        
        D_true = cdist(X_true, X_true, metric='euclidean')
        D_pred = cdist(X_pred, X_pred, metric='euclidean')
        diff = upper_triangular_values(D_true - D_pred)
        mse = np.mean(diff ** 2)
        norm_factor = np.max(D_true) - np.min(D_true)
        nrmse = np.sqrt(mse) / norm_factor if norm_factor > 0 else np.sqrt(mse)
        nrmse_values.append(nrmse)
    return np.mean(nrmse_values)

def auc_embeddings(pred_embed_list, y_embed):
    X = np.concatenate(pred_embed_list)
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    try:
        model.fit(X, y_embed)
        y_pred_prob = model.predict_proba(X)[:, 1]
        auc = adjusted_auc(y_embed, y_pred_prob)
    except ValueError:
        # If the model fails to fit, return NaN or some other indicator of failure.
        return np.nan
    return auc

def calculate_embedding_metrics(true_embed_list, pred_embed_list, y_embed):
    frob_distance = frobenius_distance(true_embed_list, pred_embed_list)
    spearman_corr = spearman_rank_correlation(true_embed_list, pred_embed_list)
    nrmse = normalized_rmse(true_embed_list, pred_embed_list)
    auc_embed = auc_embeddings(pred_embed_list, y_embed)
    return frob_distance, spearman_corr, nrmse, auc_embed
