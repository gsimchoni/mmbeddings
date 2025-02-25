import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from mmbeddings.models.lmmnn.lmmnn import run_lmmnn
from mmbeddings.models.mlp import MLP
from mmbeddings.models.embeddings import EmbeddingsMLP
from mmbeddings.models.mmbeddings import MmbeddingsDecoderPostTraining, MmbeddingsVAE
from mmbeddings.models.mmbeddings2 import MmbeddingsVAE2
from mmbeddings.models.regbeddings import RegbeddingsMLP
from mmbeddings.utils import ExpResult
from mmbeddings.metrics import calculate_embedding_metrics


class Experiment:
    def __init__(self, exp_in, exp_type, ModelClass, processing_fn=lambda x: x, plot_fn=None):
        """
        Parameters:
        exp_in : ExpInput - Input data for the experiment.
        exp_type : str - The type of experiment to run.
        ModelClass : class - The model class to use for the experiment.
        """
        self.exp_in = exp_in
        self.exp_type = exp_type
        self.X_train = exp_in.X_train
        self.X_test = exp_in.X_test
        self.y_train = exp_in.y_train
        self.y_test = exp_in.y_test
        self.n_sig2bs = exp_in.n_sig2bs
        self.model_class = ModelClass
        self.processing_fn = processing_fn
        self.plot_fn = plot_fn
        if self.exp_in.y_type == 'continuous':
            self.loss = 'mse'
            self.last_layer_activation = 'linear'
        elif self.exp_in.y_type == 'binary':
            self.loss = 'binary_crossentropy'
            self.last_layer_activation = 'sigmoid'
        else:
            raise ValueError(f'Unsupported y_type: {self.y_type}')
    
    def run(self):
        """
        Run the experiment, store the results in self.exp_res.
        """
        start = time.time()
        X_train, X_test = self.prepare_input_data()
        input_dim = self.get_input_dimension(X_train)
        model = self.model_class(self.exp_in, input_dim, self.loss, self.last_layer_activation)
        model.build()
        history = model.fit(X_train, self.y_train)
        y_pred = model.predict(X_test)
        y_pred = self.processing_fn(y_pred)
        end = time.time()
        runtime = end - start
        metric, sigmas, nll_tr, nll_te, n_epochs, n_params = model.summarize(self.y_test, y_pred, history)
        frobenius, spearman, nrmse = np.nan, np.nan, np.nan
        if self.plot_fn:
            self.plot_fn(self.y_test, y_pred.flatten())
        self.exp_res = ExpResult(metric, frobenius, spearman, nrmse, sigmas, nll_tr, nll_te, n_epochs, runtime, n_params)

    def summarize(self):
        """
        Summarize the results of the experiment.
        """
        res_summary = [self.exp_in.n_train, self.exp_in.n_test, self.exp_in.batch] +\
            [self.exp_in.pred_unknown, self.exp_in.sig2e, self.exp_in.beta_vae] +\
            list(self.exp_in.sig2bs) + list(self.exp_in.qs) +\
            [self.exp_in.k, self.exp_type, self.exp_res.metric] +\
            [self.exp_res.frobenius, self.exp_res.spearman, self.exp_res.nrmse, self.exp_res.sigmas[0]] +\
            self.exp_res.sigmas[1] + [self.exp_res.nll_tr, self.exp_res.nll_te] + \
                [self.exp_res.n_epochs, self.exp_res.time, self.exp_res.n_params]
        return res_summary
    
    def get_input_dimension(self, X_train):
        return len(self.exp_in.x_cols)


class IgnoreOHE(Experiment):
    def __init__(self, exp_in, ignore_RE, processing_fn=lambda x: x):
        if ignore_RE:
            super().__init__(exp_in, 'ignore', MLP, processing_fn)
        else:
            super().__init__(exp_in, 'ohe', MLP, processing_fn)
        self.ignore_RE = ignore_RE
    
    def process_one_hot_encoding(self, X_train, X_test, x_cols):
        z_cols = X_train.columns[X_train.columns.str.startswith('z')]
        X_train_new = X_train[x_cols]
        X_test_new = X_test[x_cols]
        for z_col in z_cols:
            X_train_ohe = pd.get_dummies(X_train[z_col], dtype='int')
            X_test_ohe = pd.get_dummies(X_test[z_col], dtype='int')
            X_test_cols_in_train = set(X_test_ohe.columns).intersection(X_train_ohe.columns)
            X_train_cols_not_in_test = set(X_train_ohe.columns).difference(X_test_ohe.columns)
            X_test_comp = pd.DataFrame(np.zeros((X_test.shape[0], len(X_train_cols_not_in_test))),
                columns=list(X_train_cols_not_in_test), dtype=np.uint8, index=X_test.index)
            if not X_test_comp.empty:
                X_test_ohe_comp = pd.concat([X_test_ohe[list(X_test_cols_in_train)], X_test_comp], axis=1)
            else:
                X_test_ohe_comp = X_test_ohe[list(X_test_cols_in_train)]
            X_test_ohe_comp = X_test_ohe_comp[X_train_ohe.columns]
            X_train_ohe.columns = list(map(lambda c: z_col + '_' + str(c), X_train_ohe.columns))
            X_test_ohe_comp.columns = list(map(lambda c: z_col + '_' + str(c), X_test_ohe_comp.columns))
            X_train_new = pd.concat([X_train_new, X_train_ohe], axis=1)
            X_test_new = pd.concat([X_test_new, X_test_ohe_comp], axis=1)
        return X_train_new, X_test_new

    def prepare_input_data(self):
        if self.ignore_RE:
            X_train, X_test = self.X_train[self.exp_in.x_cols], self.X_test[self.exp_in.x_cols]
        else:
            X_train, X_test = self.process_one_hot_encoding(self.X_train, self.X_test, self.exp_in.x_cols)
        return X_train, X_test
    
    def get_input_dimension(self, X_train):
        return X_train.shape[1]

class Embeddings(Experiment):
    def __init__(self, exp_in, processing_fn=lambda x: x, plot_fn=None, growth_model=False, l2reg_lambda=None, simulation_mode=True):
        super().__init__(exp_in, 'embeddings' if l2reg_lambda is None else 'embeddings-l2', Embeddings, processing_fn, plot_fn)
        self.growth_model = growth_model
        self.l2reg_lambda = l2reg_lambda
        self.simulation_mode = simulation_mode

    def prepare_input_data(self):
        X_train_z_cols = [self.X_train[z_col] for z_col in self.X_train.columns[self.X_train.columns.str.startswith('z')]]
        X_test_z_cols = [self.X_test[z_col] for z_col in self.X_train.columns[self.X_train.columns.str.startswith('z')]]
        X_train_input = [self.X_train[self.exp_in.x_cols]] + X_train_z_cols
        X_test_input = [self.X_test[self.exp_in.x_cols]] + X_test_z_cols
        return X_train_input, X_test_input
    
    def run(self):
        """
        Run the experiment, store the results in self.exp_res.
        """
        start = time.time()
        X_train, X_test = self.prepare_input_data()
        input_dim = self.get_input_dimension(X_train)
        model = EmbeddingsMLP(self.exp_in, input_dim, self.last_layer_activation, self.growth_model, self.l2reg_lambda)
        model.compile(loss=self.loss, optimizer='adam')
        history = model.fit_model(X_train, self.y_train)
        embeddings_list = [embed.get_weights()[0] for embed in model.encoder.embeddings]
        sig2bs_hat_list = [embeddings_list[i].var(axis=0) for i in range(len(embeddings_list))]
        y_pred = model.predict(X_test, verbose=self.exp_in.verbose, batch_size=self.exp_in.batch)
        y_pred = self.processing_fn(y_pred)
        end = time.time()
        runtime = end - start
        metric, sigmas, nll_tr, nll_te, n_epochs, n_params = model.summarize(self.y_test, y_pred, history, sig2bs_hat_list)
        frobenius, spearman, nrmse = np.nan, np.nan, np.nan
        if self.simulation_mode:
            frobenius, spearman, nrmse = calculate_embedding_metrics(self.exp_in.B_true_list, embeddings_list)
        if self.plot_fn:
            self.plot_fn(self.y_test, y_pred.flatten())
        self.exp_res = ExpResult(metric, frobenius, spearman, nrmse, sigmas, nll_tr, nll_te, n_epochs, runtime, n_params)


class REbeddings(Experiment):
    def __init__(self, exp_in, REbeddings_type, processing_fn=lambda x: x, plot_fn=None, growth_model=False, simulation_mode=True):
        super().__init__(exp_in, REbeddings_type, REbeddings, processing_fn, plot_fn)
        self.growth_model = growth_model
        self.RE_cols = self.get_RE_cols_by_prefix(self.X_train, self.exp_in.RE_cols_prefix)
        self.diverse_batches = False
        self.simulation_mode = simulation_mode
        if REbeddings_type == 'mmbeddings':
            self.model_class = MmbeddingsVAE
        elif REbeddings_type == 'regbeddings':
            self.model_class = RegbeddingsMLP
        elif REbeddings_type == 'mmbeddings-v2':
            self.model_class = MmbeddingsVAE2
    
    def run(self):
        start = time.time()
        if self.diverse_batches:
            self.diversify_batches()
        X_train, Z_train, X_test, Z_test = self.prepare_input_data()
        input_dim = self.get_input_dimension(X_train)
        model = self.model_class(self.exp_in, input_dim, self.last_layer_activation, self.growth_model)
        model.compile(optimizer='adam')
        history = model.fit_model(X_train, Z_train, self.y_train, shuffle=not self.diverse_batches)
        embeddings_list, sig2bs_hat_list = model.predict_embeddings(X_train, Z_train, self.y_train)
        if self.exp_type == 'mmbeddings':
            # uncomment to see the difference in test MSE when adding decoder post training
            # y_pred0 = model.predict_model(X_test, Z_test, embeddings_list)
            # mse0 = np.mean((self.y_test - y_pred0) ** 2)
            embeddings_list_processed = model.replicate_Bs_to_predict(Z_train, embeddings_list)
            model_post_trainer = MmbeddingsDecoderPostTraining(self.exp_in, model.decoder, self.exp_type)
            model_post_trainer.compile(optimizer='adam', loss='mse')
            model_post_trainer.fit_model(X_train, Z_train, embeddings_list_processed, self.y_train)
        y_pred = model.predict_model(X_test, Z_test, embeddings_list)
        y_pred = self.processing_fn(y_pred)
        losses_tr = model.evaluate_model(X_train, Z_train, self.y_train)
        losses_te = model.evaluate_model(X_test, Z_test, self.y_test)
        end = time.time()
        runtime = end - start
        metric, sigmas, nll_tr, nll_te, n_epochs, n_params = model.summarize(self.y_test, y_pred, sig2bs_hat_list, losses_tr, losses_te, history)
        frobenius, spearman, nrmse = np.nan, np.nan, np.nan
        if self.simulation_mode:
            frobenius, spearman, nrmse = calculate_embedding_metrics(self.exp_in.B_true_list, embeddings_list)
        if self.plot_fn:
            self.plot_fn(self.y_test, y_pred.flatten())
        if self.diverse_batches:
            self.undiversify_batches()
        self.exp_res = ExpResult(metric, frobenius, spearman, nrmse, sigmas, nll_tr, nll_te, n_epochs, runtime, n_params)

    def prepare_input_data(self):
        X_train, Z_train = self.prepare_input_data_single_set(self.X_train)
        X_test, Z_test = self.prepare_input_data_single_set(self.X_test)
        return X_train, Z_train, X_test, Z_test

    def get_RE_cols_by_prefix(self, df, prefix):
        RE_cols = list(df.columns[df.columns.str.startswith(prefix)])
        return RE_cols
    
    def prepare_input_data_single_set(self, X):
        X_train = X[self.exp_in.x_cols].copy()
        Z_train = [X[RE_col].copy() for RE_col in self.RE_cols]
        return X_train, Z_train
    
    def diversify_batches(self):
        self.orig_idx_train = self.X_train.index
        self.X_train['idx'] = self.X_train.index
        grouped = self.X_train.groupby(self.RE_cols)
        interleaved_indices = []
        max_cluster_size = max(grouped.size())
        for i in range(max_cluster_size):
            sampled = grouped.nth(i).dropna()  # Take the i-th element from each cluster if available
            interleaved_indices.extend(sampled['idx'].tolist())
        self.X_train = self.X_train.loc[interleaved_indices].drop(columns=['idx'])
        self.y_train = self.y_train.reindex(self.X_train.index)

    def undiversify_batches(self):
        self.X_train = self.X_train.reindex(self.orig_idx_train)
        self.y_train = self.y_train.reindex(self.orig_idx_train)


class LMMNN(Experiment):
    def __init__(self, exp_in, processing_fn=lambda x: x):
        super().__init__(exp_in, 'lmmnn', LMMNN, processing_fn)
    
    def run(self):
        start = time.time()
        (q_spatial, mode, y_type, n_sig2bs_spatial, est_cors, dist_matrix,
         spatial_embed_neurons, Z_non_linear, shuffle, sample_n_train) = self.get_init_vals()
        y_pred, sigmas, rhos, n_epochs, nll_tr, nll_te, y_pred_no_re, n_params = run_lmmnn(
            self.X_train, self.X_test, self.y_train, self.y_test, self.exp_in.qs,
            q_spatial, self.exp_in.x_cols, self.exp_in.batch, self.exp_in.epochs,
            self.exp_in.patience, self.exp_in.n_neurons, self.exp_in.dropout,
            self.exp_in.activation, mode, y_type, self.n_sig2bs, n_sig2bs_spatial,
            est_cors, dist_matrix, spatial_embed_neurons, self.exp_in.verbose,
            Z_non_linear, self.exp_in.Z_embed_dim_pct, self.exp_in.log_params,
            self.exp_in.k, shuffle, sample_n_train, self.exp_in.B_true_list)
        y_pred = self.processing_fn(y_pred)
        end = time.time()
        runtime = end - start
        if y_type == 'continuous':
            metric = np.mean((y_pred - self.y_test)**2)
        elif y_type == 'binary':
            metric = roc_auc_score(self.y_test, y_pred)
        else:
            raise ValueError(f'Unsupported y_type: {y_type}')
        frobenius, spearman, nrmse = np.nan, np.nan, np.nan
        self.exp_res = ExpResult(metric, frobenius, spearman, nrmse, sigmas, nll_tr, nll_te, n_epochs, runtime, n_params)

    def get_init_vals(self):
        q_spatial = None
        mode = 'categorical'
        y_type = self.exp_in.y_type
        n_sig2bs_spatial = 0
        est_cors = []
        dist_matrix = None
        spatial_embed_neurons = []
        Z_non_linear = False
        shuffle = True
        sample_n_train=10000
        return q_spatial,mode,y_type,n_sig2bs_spatial,est_cors,dist_matrix,spatial_embed_neurons,Z_non_linear,shuffle,sample_n_train