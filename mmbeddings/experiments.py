import time
import numpy as np
import pandas as pd
from mmbeddings.models import MLP, EmbeddingsMLP, MmbeddingsVAE
from mmbeddings.utils import ExpResult


class Experiment:
    def __init__(self, exp_in, exp_type, ModelClass):
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
    
    def run(self):
        """
        Run the experiment, store the results in self.exp_res.
        """
        start = time.time()
        X_train, X_test = self.prepare_input_data()
        input_dim = self.get_input_dimension(X_train)
        model = self.model_class(self.exp_in, input_dim)
        model.build()
        history = model.fit(X_train, self.y_train)
        y_pred = model.predict(X_test)
        end = time.time()
        runtime = end - start
        mse, sigmas, nll_tr, nll_te, n_epochs = model.summarize(self.y_test, y_pred, history)
        self.exp_res = ExpResult(mse, sigmas, nll_tr, nll_te, n_epochs, runtime)

    def summarize(self):
        """
        Summarize the results of the experiment.
        """
        res_summary = [self.exp_in.N, self.exp_in.test_size, self.exp_in.batch, self.exp_in.pred_unknown, self.exp_in.sig2e] +\
            list(self.exp_in.sig2bs) + list(self.exp_in.qs) +\
            [self.exp_in.k, self.exp_type, self.exp_res.mse, self.exp_res.sigmas[0]] +\
            self.exp_res.sigmas[1] + [self.exp_res.nll_tr, self.exp_res.nll_te, self.exp_res.n_epochs, self.exp_res.time]
        return res_summary
    
    def get_input_dimension(self, X_train):
        return len(self.exp_in.x_cols)


class IgnoreOHE(Experiment):
    def __init__(self, exp_in, ignore_RE):
        if ignore_RE:
            super().__init__(exp_in, 'ignore', MLP)
        else:
            super().__init__(exp_in, 'ohe', MLP)
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
    def __init__(self, exp_in, growth_model=False):
        super().__init__(exp_in, 'embeddings', Embeddings)
        self.growth_model = growth_model

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
        model = EmbeddingsMLP(self.exp_in, input_dim, self.growth_model)
        model.compile(loss='mse', optimizer='adam')
        history = model.fit_model(X_train, self.y_train)
        y_pred = model.predict(X_test)
        end = time.time()
        runtime = end - start
        mse, sigmas, nll_tr, nll_te, n_epochs = model.summarize(self.y_test, y_pred, history)
        self.exp_res = ExpResult(mse, sigmas, nll_tr, nll_te, n_epochs, runtime)

class Mmbeddings(Experiment):
    def __init__(self, exp_in, growth_model=False):
        super().__init__(exp_in, 'mmbeddings', Mmbeddings)
        self.growth_model = growth_model
        self.RE_cols = self.get_RE_cols_by_prefix(self.X_train, self.exp_in.RE_cols_prefix)
    
    def run(self):
        start = time.time()
        X_train, Z_train, X_test, Z_test = self.prepare_input_data()
        input_dim = self.get_input_dimension(X_train)
        model = MmbeddingsVAE(self.exp_in, input_dim, self.growth_model)
        model.compile(optimizer='adam')
        history = model.fit_model(X_train, Z_train, self.y_train)
        mmbeddings_list, sig2bs_hat_list = model.predict_mmbeddings(X_train, Z_train, self.y_train)
        y_pred = model.predict(X_test, Z_test, mmbeddings_list)
        losses_tr = model.evaluate_model(X_train, Z_train, self.y_train)
        losses_te = model.evaluate_model(X_test, Z_test, self.y_test)
        end = time.time()
        runtime = end - start
        mse, sigmas, nll_tr, nll_te, n_epochs = model.summarize(self.y_test, y_pred, sig2bs_hat_list, losses_tr, losses_te, history)
        self.exp_res = ExpResult(mse, sigmas, nll_tr, nll_te, n_epochs, runtime)

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