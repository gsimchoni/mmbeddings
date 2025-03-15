from dataclasses import dataclass, field
from typing import List, Optional


import numpy as np
import scipy.sparse as sparse
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

@dataclass
class ExpResult:
    metrics: List[float]
    nll_tr: float
    nll_te: float
    n_epochs: int
    time: float
    n_params: int
    frobenius: float = field(default=np.nan)
    spearman: float = field(default=np.nan)
    nrmse: float = field(default=np.nan)
    auc_embed: float = field(default=np.nan)
    sigmas: List[float] = field(default_factory=list)

@dataclass
class ExpData:
    X_train: any
    X_test: any
    y_train: any
    y_test: any
    x_cols: List[str]
    B_true_list: List[np.ndarray]
    y_embeddings: np.ndarray

@dataclass
class ExpInput:
    X_train: any
    X_test: any
    y_train: any
    y_test: any
    x_cols: List[str]
    B_true_list: List[np.ndarray]
    y_embeddings: np.ndarray
    
    n_train: int
    n_test: int
    pred_unknown: bool
    qs: List[int]
    d: int
    sig2e: float
    sig2bs: List[float]
    y_type: str
    k: int
    batch: int
    epochs: int
    patience: int
    Z_embed_dim_pct: float
    n_sig2bs: int
    verbose: bool
    n_neurons: int
    n_neurons_encoder: int = field(default=None)
    dropout: float = 0.0
    activation: str = "relu"
    RE_cols_prefix: str = "z"
    re_sig2b_prior: float = 1.0
    beta_vae: float = 1.0
    hashing_bins: int = 2**10
    log_params: bool = False
    mmbeddings_post_training: bool = True
    epochs_post_training: Optional[int] = None
    patience_post_training: Optional[int] = None

    def __post_init__(self):
        """Handle defaults that depend on other fields."""
        if self.n_neurons_encoder is None:
            self.n_neurons_encoder = self.n_neurons
        if self.epochs_post_training is None:
            self.epochs_post_training = self.epochs
        if self.patience_post_training is None:
            self.patience_post_training = self.patience

class Count:
    # Class to generate a sequence of numbers  
    curr = 0

    def __init__(self, startWith=None):
        if startWith is not None:
            Count.curr = startWith - 1

    def gen(self):
        while True:
            Count.curr += 1
            yield Count.curr

def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(
        vec_size, vec_max), dtype=np.uint8)
    return Z

def get_cov_mat(sig2bs, rhos, est_cors):
    cov_mat = np.zeros((len(sig2bs), len(sig2bs)))
    for k in range(len(sig2bs)):
        for j in range(len(sig2bs)):
            if k == j:
                cov_mat[k, j] = sig2bs[k]
            else:
                rho_symbol = ''.join(map(str, sorted([k, j])))
                if rho_symbol in est_cors:
                    rho = rhos[est_cors.index(rho_symbol)]
                else:
                    rho = 0
                cov_mat[k, j] = rho * np.sqrt(sig2bs[k]) * np.sqrt(sig2bs[j])
    return cov_mat

def adjusted_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return max(auc, 1 - auc)  # Flip if necessary

def adjusted_log_loss(y_true, y_pred):
    flipped_pred = 1 - y_pred  # Flip probabilities
    loss = log_loss(y_true, y_pred)
    flipped_loss = log_loss(y_true, flipped_pred)
    return min(loss, flipped_loss)  # Take the best alignment

def adjust_accuracy(y_true, y_pred):
    flipped_pred = 1 - y_pred  # Flip probabilities
    acc = accuracy_score(y_true, y_pred > 0.5)
    flipped_acc = accuracy_score(y_true, flipped_pred > 0.5)
    return max(acc, flipped_acc)  # Take the best alignment

def evaluate_predictions(y_type, y_test, y_pred):
        if y_type == 'continuous':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = [mse, mae, r2]
        elif y_type == 'binary':
            auc = adjusted_auc(y_test, y_pred)
            logloss = adjusted_log_loss(y_test, y_pred)
            accuracy = adjust_accuracy(y_test, y_pred)
            metrics = [auc, logloss, accuracy]
        else:
            raise ValueError(f'Unsupported y_type: {y_type}')
        return metrics