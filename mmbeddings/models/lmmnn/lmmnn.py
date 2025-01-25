import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Reshape
from tensorflow.keras.models import Model

from mmbeddings.models.lmmnn.lmmnll import LMMNLL
from mmbeddings.models.lmmnn.calc_b_hat import calc_b_hat
from mmbeddings.utils import get_dummies

def add_layers_functional(X_input, n_neurons, dropout, activation, input_dim):
    if len(n_neurons) > 0:
        x = Dense(n_neurons[0], input_dim=input_dim, activation=activation)(X_input)
        if dropout is not None and len(dropout) > 0:
            x = Dropout(dropout[0])(x)
        for i in range(1, len(n_neurons) - 1):
            x = Dense(n_neurons[i], activation=activation)(x)
            if dropout is not None and len(dropout) > i:
                x = Dropout(dropout[i])(x)
        if len(n_neurons) > 1:
            x = Dense(n_neurons[-1], activation=activation)(x)
        return x
    return X_input

class LogEstParams(Callback):
    def __init__(self, idx, exp_type=0):
        super(LogEstParams, self).__init__()
        self.idx = idx
        self.exp_type = exp_type

    def on_epoch_end(self, epoch, logs):
        sig2e_est, sig2bs_est, rhos_est, weibull_est = self.model.layers[-1].get_vars()
        logs['exp_type'] = self.exp_type
        logs['experiment'] = self.idx
        logs['sig2e_est'] = sig2e_est
        for k, sig2b_est in enumerate(sig2bs_est):
            logs['sig2b_est' + str(k)] = sig2b_est
        for k, rho_est in enumerate(rhos_est):
            logs['rho_est' + str(k)] = rho_est
        for k, weibull_est in enumerate(weibull_est):
            logs['weibull_est' + str(k)] = weibull_est

def get_callbacks(patience, epochs, Z_non_linear, mode, log_params, idx, exp_type_num):
    patience = epochs if patience is None else patience
    if Z_non_linear and mode == 'categorical':
        # in complex scenarios such as non-linear g(Z) consider training "more", until var components norm has converged
        # callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience)]
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    else:
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    if log_params:
        callbacks.extend([LogEstParams(idx, exp_type_num), CSVLogger('res_params.csv', append=True)])
    return callbacks

def get_sig2_ests(mode, model):
    sig2e_est, sig2b_ests, rho_ests, weibull_ests = model.layers[-1].get_vars()
    if mode in ['spatial', 'spatial_embedded']:
        sig2b_spatial_ests = sig2b_ests
        sig2b_ests = []
    elif mode == 'spatial_categorical':
        sig2b_spatial_ests = sig2b_ests[:2]
        sig2b_ests = sig2b_ests[2:]
    else:
        sig2b_spatial_ests = []
    return sig2e_est, sig2b_ests, rho_ests, weibull_ests, sig2b_spatial_ests

def run_lmmnn(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
        mode, y_type, n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons,
        verbose=False, Z_non_linear=False, Z_embed_dim_pct=10, log_params=False, idx=0, shuffle=False, sample_n_train=10000, b_true=None):
    if mode in ['spatial', 'spatial_embedded', 'spatial_categorical']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2']]
    if mode == 'survival':
        x_cols = [x_col for x_col in x_cols if x_col not in ['C0']]
    # dmatrix_tf = tf.constant(dist_matrix)
    dmatrix_tf = dist_matrix
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    if mode in ['categorical', 'spatial', 'spatial_categorical'] or y_type == 'binary':
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        Z_inputs = []
        if mode in ['spatial']:
            n_sig2bs_init = n_sig2bs_spatial
            n_RE_inputs = 1
        elif mode == 'spatial_categorical':
            n_sig2bs_init = n_sig2bs_spatial + len(qs)
            n_RE_inputs = 1 + len(qs)
        else:
            n_sig2bs_init = len(qs)
            n_RE_inputs = len(qs)
        for _ in range(n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
    elif mode == 'longitudinal':
        z_cols = ['z0', 't']
        n_RE_inputs = 2
        n_sig2bs_init = n_sig2bs
        Z_input = Input(shape=(1,), dtype=tf.int64)
        t_input = Input(shape=(1,))
        Z_inputs = [Z_input, t_input]
    elif mode == 'spatial_embedded':
        Z_inputs = [Input(shape=(2,))]
        n_sig2bs_init = 1
    elif mode == 'survival':
        z_cols = ['z0', 'C0']
        Z_input = Input(shape=(1,), dtype=tf.int64)
        event_input = Input(shape=(1,))
        Z_inputs = [Z_input, event_input]
        n_sig2bs_init = 1
    
    out_hidden = add_layers_functional(X_input, n_neurons, dropout, activation, X_train[x_cols].shape[1])
    y_pred_output = Dense(1)(out_hidden)
    if Z_non_linear and (mode in ['categorical', 'survival'] or y_type == 'binary'):
        Z_nll_inputs = []
        ls = []
        for k, q in enumerate(qs):
            l = int(q * Z_embed_dim_pct / 100.0)
            Z_embed = Embedding(q, l, input_length=1, name='Z_embed' + str(k))(Z_inputs[k])
            Z_embed = Reshape(target_shape=(l, ))(Z_embed)
            Z_nll_inputs.append(Z_embed)
            ls.append(l)
    elif mode == 'spatial_embedded':
        Z_embed = add_layers_functional(Z_inputs[0], spatial_embed_neurons, dropout=None, activation='relu', input_dim=2)
        Z_nll_inputs = [Z_embed]
        ls = [spatial_embed_neurons[-1]]
        Z_non_linear = True
    else:
        Z_nll_inputs = Z_inputs
        ls = None
    sig2bs_init = np.ones(n_sig2bs_init, dtype=np.float32)
    rhos_init = np.zeros(len(est_cors), dtype=np.float32)
    weibull_init = np.ones(2, dtype=np.float32)
    nll = LMMNLL(mode, y_type, 1.0, sig2bs_init, rhos_init, weibull_init, est_cors, Z_non_linear, dmatrix_tf)(
        y_true_input, y_pred_output, Z_nll_inputs)
    model = Model(inputs=[X_input, y_true_input] + Z_inputs, outputs=nll)

    model.compile(optimizer='adam')

    callbacks = get_callbacks(patience, epochs, Z_non_linear, mode, log_params, idx, exp_type_num=0)

    if not Z_non_linear:
        orig_idx_train = X_train.index
        orig_idx_test = X_test.index
        X_train = X_train.sort_values(by=z_cols)
        y_train = y_train.reindex(X_train.index)
        X_test = X_test.sort_values(by=z_cols)
        y_test = y_test.reindex(X_test.index)
    if mode == 'spatial_embedded':
        X_train_z_cols = [X_train[['D1', 'D2']]]
        X_test_z_cols = [X_test[['D1', 'D2']]]
    else:
        X_train_z_cols = [X_train[z_col] for z_col in z_cols]
        X_test_z_cols = [X_test[z_col] for z_col in z_cols]
    history = model.fit([X_train[x_cols], y_train] + X_train_z_cols, None,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=verbose, shuffle=shuffle)
    nll_tr = model.evaluate([X_train[x_cols], y_train] + X_train_z_cols, batch_size=batch_size, verbose=verbose)
    nll_te = model.evaluate([X_test[x_cols], y_test] + X_test_z_cols, batch_size=batch_size, verbose=verbose)
    X_train = X_train.reindex(orig_idx_train)
    y_train = y_train.reindex(orig_idx_train)
    X_test = X_test.reindex(orig_idx_test)
    y_test = y_test.reindex(orig_idx_test)

    sig2e_est, sig2b_ests, rho_ests, weibull_ests, sig2b_spatial_ests = get_sig2_ests(mode, model)

    y_pred_tr = model.predict(
        [X_train[x_cols], y_train] + X_train_z_cols, verbose=verbose).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, qs, q_spatial, sig2e_est, sig2b_ests, sig2b_spatial_ests,
                Z_non_linear, model, ls, mode, rho_ests, est_cors, dist_matrix, weibull_ests, y_type, sample_n_train)
    dummy_y_test = np.random.normal(size=y_test.shape)
    if y_type == 'binary':
        dummy_y_test = pd.Series(np.random.binomial(1, 0.5, size=y_test.shape))
    if mode in ['categorical', 'spatial', 'spatial_categorical'] or y_type == 'binary':
        if Z_non_linear or (len(qs) > 1 and y_type == 'continuous') or mode == 'spatial_categorical':
            delta_loc = 0
            if mode == 'spatial_categorical':
                delta_loc = 1
            Z_tests = []
            for k, q in enumerate(qs):
                Z_test = get_dummies(X_test['z' + str(k + delta_loc)], q)
                if Z_non_linear:
                    W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                    Z_test = Z_test @ W_est
                Z_tests.append(Z_test)
            if Z_non_linear:
                Z_test = np.hstack(Z_tests)
            else:
                Z_test = sparse.hstack(Z_tests)
            if mode == 'spatial_categorical':
                Z_test = sparse.hstack([Z_test, get_dummies(X_test['z0'], q_spatial)])
            y_pred_no_re = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0])
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
        else:
            # if model input is that large, this 2nd call to predict may cause OOM due to GPU memory issues
            # if that is the case use tf.convert_to_tensor() explicitly with a call to model() without using predict() method
            # y_pred = model([tf.convert_to_tensor(X_test[x_cols]), tf.convert_to_tensor(dummy_y_test), tf.convert_to_tensor(X_test_z_cols[0])], training=False).numpy().reshape(
            #     X_test.shape[0]) + b_hat[X_test['z0']]
            y_pred_no_re = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0])
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + b_hat[X_test['z0']]
        if y_type == 'binary':
            y_pred = np.exp(y_pred)/(1 + np.exp(y_pred))
    elif mode == 'longitudinal':
        q = qs[0]
        Z0 = get_dummies(X_test['z0'], q)
        t = X_test['t'].values
        N = X_test.shape[0]
        Z_list = [Z0]
        for k in range(1, len(sig2b_ests)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Z_test = sparse.hstack(Z_list)
        y_pred_no_re = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0])
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
    elif mode == 'spatial_embedded':
        last_layer = Model(inputs = model.input[2], outputs = model.layers[-2].output)
        gZ_test = last_layer.predict(X_test_z_cols, verbose=verbose)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + gZ_test @ b_hat
        sig2b_spatial_ests = np.concatenate([sig2b_spatial_ests, [np.nan]])
    elif mode == 'survival':
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0])
        y_pred = y_pred + np.log(b_hat[X_test['z0']])
    return y_pred, (sig2e_est, list(sig2b_ests), list(sig2b_spatial_ests)), list(rho_ests), len(history.history['loss']), nll_tr, nll_te, y_pred_no_re