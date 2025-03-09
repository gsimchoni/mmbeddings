from dataclasses import dataclass, field
from typing import List, Optional


import numpy as np
import scipy.sparse as sparse

@dataclass
class ExpResult:
    metric: float
    nll_tr: float
    nll_te: float
    n_epochs: int
    time: float
    n_params: int
    frobenius: float = field(default=np.nan)
    spearman: float = field(default=np.nan)
    nrmse: float = field(default=np.nan)
    sigmas: List[float] = field(default_factory=list)

@dataclass
class ExpData:
    X_train: any
    X_test: any
    y_train: any
    y_test: any
    x_cols: List[str]
    B_true_list: List[np.ndarray]

@dataclass
class ExpInput:
    X_train: any
    X_test: any
    y_train: any
    y_test: any
    x_cols: List[str]
    B_true_list: List[np.ndarray]
    
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