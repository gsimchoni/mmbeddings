from collections import namedtuple

import numpy as np
import scipy.sparse as sparse

ExpResult = namedtuple('ExpResult', ['metric', 'frobenius', 'spearman', 'nrmse', 'sigmas', 'nll_tr', 'nll_te', 'n_epochs', 'time', 'n_params'])

ExpData = namedtuple('ExpData', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols', 'B_true_list'])

ExpInput = namedtuple('ExpInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols', 'B_true_list',
                                   'n_train', 'n_test', 'pred_unknown', 'qs', 'd', 'sig2e',
                                   'sig2bs', 'y_type', 'k', 'batch', 'epochs', 'patience',
                                   'Z_embed_dim_pct', 'n_sig2bs', 'verbose',
                                   'n_neurons', 'n_neurons_encoder', 'dropout',
                                   'activation', 'RE_cols_prefix',
                                   're_sig2b_prior', 'beta_vae', 'hashing_bins',
                                   'log_params', 'mmbeddings_post_training',
                                   'epochs_post_training', 'patience_post_training'
                                   ])

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