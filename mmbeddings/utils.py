from collections import namedtuple

import numpy as np
import scipy.sparse as sparse

ExpResult = namedtuple('ExpResult', ['mse', 'sigmas', 'nll_tr', 'nll_te', 'n_epochs', 'time'])

ExpData = namedtuple('ExpData', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols'])

ExpInput = namedtuple('ExpInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols',
                                   'N', 'test_size', 'pred_unknown', 'qs', 'd', 'sig2e',
                                   'sig2bs', 'k', 'batch', 'epochs', 'patience',
                                   'Z_embed_dim_pct', 'n_sig2bs', 'verbose',
                                   'n_neurons', 'dropout', 'activation', 'RE_cols_prefix',
                                   're_sig2b_prior', 'beta_vae', 'log_params',
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
