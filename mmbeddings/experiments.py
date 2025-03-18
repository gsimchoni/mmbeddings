import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mmbeddings.models.lmmnn.lmmnn import run_lmmnn
from mmbeddings.models.mlp import MLP
from mmbeddings.models.embeddings import EmbeddingsMLP, HashingMLP
from mmbeddings.models.mmbeddings import MmbeddingsDecoderPostTraining, MmbeddingsVAE
from mmbeddings.models.mmbeddings2 import MmbeddingsVAE2
from mmbeddings.models.regbeddings import RegbeddingsMLP
from mmbeddings.models.tabtransformer import TabTransformerModel
from mmbeddings.models.tf_tabnet.tabnet import TabNetModel
from mmbeddings.models.ncf import NCFModel
from mmbeddings.utils import ExpResult, evaluate_predictions
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
        model = self.model_class(self.exp_in, input_dim, self.last_layer_activation)
        model.compile(loss=self.loss, optimizer='adam')
        history = model.fit_model(X_train, self.y_train)
        sig2bs_hat_list = [np.nan] * len(self.exp_in.qs)
        y_pred = model.predict(X_test, verbose=self.exp_in.verbose, batch_size=self.exp_in.batch)
        y_pred = self.processing_fn(y_pred)
        end = time.time()
        runtime = end - start
        metrics, sigmas, nll_tr, nll_te, n_epochs, n_params = model.summarize(self.y_test, y_pred, history, sig2bs_hat_list)
        if self.plot_fn:
            self.plot_fn(self.y_test, y_pred.flatten())
        self.exp_res = ExpResult(metrics=metrics, sigmas=sigmas, nll_tr=nll_tr,
                                 nll_te=nll_te, n_epochs=n_epochs, time=runtime, n_params=n_params)
    
    def get_input_dimension(self, X_train):
        return len(self.exp_in.x_cols)


class PrecomputedEmbeddingExperiment(Experiment):
    def __init__(self, exp_in, encoding_type, processing_fn=lambda x: x, plot_fn=None):
        super().__init__(exp_in, encoding_type, MLP, processing_fn, plot_fn)
        self.encoding_type = encoding_type
    
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
    
    def process_mean_encoding(self, X_train, X_test, x_cols):
        """
        For each categorical feature (z columns), compute the mean of each numerical feature (x_cols)
        in the training data, then map these means to both training and test sets.
        
        Args:
            X_train (pd.DataFrame): Training data with numerical features (x_cols) and categorical features (columns starting with 'z').
            X_test (pd.DataFrame): Test data with similar columns.
            x_cols (list): List of names of numerical features.
        
        Returns:
            Tuple of pd.DataFrame: (X_train_new, X_test_new) with the original x_cols and new mean-encoded features.
        """
        # Identify the categorical columns (those starting with "z")
        z_cols = X_train.columns[X_train.columns.str.startswith('z')]
        
        # Start with the original numerical features.
        X_train_new = X_train[x_cols].copy()
        X_test_new = X_test[x_cols].copy()
        
        for z in z_cols:
            # Compute the means of the x_cols for each group in the training data.
            group_means = X_train.groupby(z)[x_cols].mean()
            # Rename the columns to reflect the mean encoding (e.g. "z0_mean_feature1")
            group_means.rename(columns=lambda c: f"{z}_mean_{c}", inplace=True)
            
            # Map the computed group means back onto the training data.
            for new_col in group_means.columns:
                X_train_new[new_col] = X_train[z].map(group_means[new_col])
                # For test data: if a category is not present in training, fill with 0.
                X_test_new[new_col] = X_test[z].map(group_means[new_col]).fillna(0)
        
        return X_train_new, X_test_new

    def process_pca_encoding(self, X_train, X_test, x_cols, n_components=5):
        """
        For each categorical feature (z columns), compute the group means over the numerical features (x_cols)
        in the training data, perform PCA on the resulting (q x p) matrix to reduce it to (q x d) where d = n_components,
        then map these PCA features back to both training and test data.
        
        For any category in the test set that is not present in the training set, the PCA features are filled with 0.
        
        Args:
            X_train (pd.DataFrame): Training data containing numerical features (x_cols) and categorical features (z_cols).
            X_test (pd.DataFrame): Test data with similar columns.
            x_cols (list): List of names of numerical features.
            n_components (int): The number of PCA components to keep (i.e. the target embedding dimension).
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The transformed training and test dataframes including the PCA-encoded features.
        """
        # Identify categorical columns (those starting with "z")
        z_cols = X_train.columns[X_train.columns.str.startswith('z')]
        
        # Start with the original numerical features
        X_train_new = X_train[x_cols].copy()
        X_test_new = X_test[x_cols].copy()
        
        for z in z_cols:
            # Compute the means of the numerical features grouped by the categorical feature in training data.
            group_means = X_train.groupby(z)[x_cols].mean()  # shape (q, p)
            
            # Fit PCA on this group-means matrix
            pca = PCA(n_components=n_components)
            group_pca = pca.fit_transform(group_means)  # shape becomes (q, n_components)
            
            # Create a DataFrame mapping each category to its PCA components
            pca_df = pd.DataFrame(group_pca, index=group_means.index, 
                                columns=[f"{z}_pca_{i}" for i in range(n_components)])
            
            # Map the PCA components back to the training data.
            for i in range(n_components):
                new_col = f"{z}_pca_{i}"
                X_train_new[new_col] = X_train[z].map(pca_df[new_col])
                # For test data: if the category is not present in training, assign 0.
                X_test_new[new_col] = X_test[z].map(pca_df[new_col]).fillna(0)
        
        return X_train_new, X_test_new
    
    def prepare_input_data(self):
        if self.encoding_type == 'ignore':
            X_train, X_test = self.X_train[self.exp_in.x_cols], self.X_test[self.exp_in.x_cols]
        elif self.encoding_type == 'ohe':
            X_train, X_test = self.process_one_hot_encoding(self.X_train, self.X_test, self.exp_in.x_cols)
        elif self.encoding_type == 'mean-encoding':
            X_train, X_test = self.process_mean_encoding(self.X_train, self.X_test, self.exp_in.x_cols)
        elif self.encoding_type == 'pca-encoding':
            X_train, X_test = self.process_pca_encoding(self.X_train, self.X_test, self.exp_in.x_cols)
        else:
            raise ValueError(f'Unsupported encoding type: {self.encoding_type}')
        return X_train, X_test
    
    def get_input_dimension(self, X_train):
        return X_train.shape[1]

class Embeddings(Experiment):
    def __init__(self, exp_in, processing_fn=lambda x: x, plot_fn=None,
                 growth_model=False, l2reg_lambda=None, simulation_mode=True):
        exp_name = self.determine_exp_name(l2reg_lambda)
        super().__init__(exp_in, exp_name, Embeddings, processing_fn, plot_fn)
        self.growth_model = growth_model
        self.l2reg_lambda = l2reg_lambda
        self.simulation_mode = simulation_mode

    def determine_exp_name(self, l2reg_lambda):
        if l2reg_lambda is None:
            exp_name = 'embeddings'
        else:
            exp_name = 'embeddings-l2'
        return exp_name

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
        model = EmbeddingsMLP(self.exp_in, input_dim, self.last_layer_activation,
                              self.growth_model, self.l2reg_lambda)
        model.compile(loss=self.loss, optimizer='adam')
        history = model.fit_model(X_train, self.y_train)
        embeddings_list = [embed.get_weights()[0] for embed in model.encoder.embeddings]
        sig2bs_hat_list = [embeddings_list[i].var(axis=0) for i in range(len(embeddings_list))]
        y_pred = model.predict(X_test, verbose=self.exp_in.verbose, batch_size=self.exp_in.batch)
        y_pred = self.processing_fn(y_pred)
        end = time.time()
        runtime = end - start
        metrics, sigmas, nll_tr, nll_te, n_epochs, n_params = model.summarize(self.y_test, y_pred, history, sig2bs_hat_list)
        frobenius, spearman, nrmse, auc_embed = np.nan, np.nan, np.nan, np.nan
        if self.simulation_mode:
            frobenius, spearman, nrmse, auc_embed = calculate_embedding_metrics(self.exp_in.B_true_list, embeddings_list, self.exp_in.y_embeddings)
        if self.plot_fn:
            self.plot_fn(self.y_test, y_pred.flatten())
        self.exp_res = ExpResult(metrics=metrics, sigmas=sigmas, nll_tr=nll_tr,
                                 nll_te=nll_te, n_epochs=n_epochs, time=runtime, n_params=n_params,
                                 frobenius=frobenius, spearman=spearman, nrmse=nrmse, auc_embed=auc_embed)


class REbeddings(Experiment):
    def __init__(self, exp_in, REbeddings_type, processing_fn=lambda x: x, plot_fn=None, growth_model=False, cf=False, simulation_mode=True):
        super().__init__(exp_in, REbeddings_type, REbeddings, processing_fn, plot_fn)
        self.growth_model = growth_model
        self.cf = cf
        self.RE_cols = self.get_RE_cols_by_prefix(self.X_train, self.exp_in.RE_cols_prefix)
        self.diverse_batches = False
        self.simulation_mode = simulation_mode
        self.evaluate = False
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
        model = self.model_class(self.exp_in, input_dim, self.last_layer_activation, self.growth_model, self.cf)
        model.compile(optimizer='adam')
        history = model.fit_model(X_train, Z_train, self.y_train, shuffle=not self.diverse_batches)
        embeddings_list, sig2bs_hat_list = model.predict_embeddings(X_train, Z_train, self.y_train)
        metrics_pre_post = [np.nan]
        if self.exp_type == 'mmbeddings' and self.exp_in.mmbeddings_post_training:
            y_pred = model.predict_model(X_test, Z_test, embeddings_list)
            metrics_pre_post = evaluate_predictions(self.exp_in.y_type, self.y_test, y_pred)
            if self.exp_in.verbose:
                print(f'MSE/AUC before post training: {metrics[0]}')
            embeddings_list_processed = model.replicate_Bs_to_predict(Z_train, embeddings_list)
            model_post_trainer = MmbeddingsDecoderPostTraining(self.exp_in, model.decoder, self.exp_type)
            model_post_trainer.compile(optimizer='adam', loss='mse')
            model_post_trainer.fit_model(X_train, Z_train, embeddings_list_processed, self.y_train)
        y_pred = model.predict_model(X_test, Z_test, embeddings_list)
        y_pred = self.processing_fn(y_pred)
        losses_tr, losses_te = [np.nan], [np.nan]
        if self.evaluate:
            losses_tr = model.evaluate_model(X_train, Z_train, self.y_train)
            losses_te = model.evaluate_model(X_test, Z_test, self.y_test)
        end = time.time()
        runtime = end - start
        metrics, sigmas, nll_tr, nll_te, n_epochs, n_params = model.summarize(self.y_test, y_pred, sig2bs_hat_list, losses_tr, losses_te, history)
        frobenius, spearman, nrmse, auc_embed = np.nan, np.nan, np.nan, np.nan
        if self.simulation_mode:
            frobenius, spearman, nrmse, auc_embed = calculate_embedding_metrics(self.exp_in.B_true_list, embeddings_list, self.exp_in.y_embeddings)
        if self.plot_fn:
            self.plot_fn(self.y_test, y_pred.flatten())
        if self.diverse_batches:
            self.undiversify_batches()
        self.exp_res = ExpResult(metrics=metrics, sigmas=sigmas, nll_tr=nll_tr,
                                 nll_te=nll_te, n_epochs=n_epochs, time=runtime, n_params=n_params,
                                 frobenius=frobenius, spearman=spearman, nrmse=nrmse, auc_embed=auc_embed,
                                 metric_pre_post = metrics_pre_post[0])

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
    def __init__(self, exp_in, processing_fn=lambda x: x, plot_fn=None):
        super().__init__(exp_in, 'lmmnn', LMMNN, processing_fn, plot_fn)
    
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
        metrics = evaluate_predictions(self.exp_in.y_type, self.y_test, y_pred)
        if self.plot_fn:
            self.plot_fn(self.y_test, y_pred.flatten())
        self.exp_res = ExpResult(metrics=metrics, sigmas=sigmas, nll_tr=nll_tr,
                                 nll_te=nll_te, n_epochs=n_epochs, time=runtime, n_params=n_params)

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


class TrainableEmbeddingExperiment(Experiment):
    def __init__(self, exp_in, net_type, processing_fn=lambda x: x, plot_fn=None):
        super().__init__(exp_in, net_type, TrainableEmbeddingExperiment, processing_fn, plot_fn)
        self.model_class = self.determine_model_class(net_type)

    def determine_model_class(self, net_type):
        if net_type == 'tabnet':
            return TabNetModel
        elif net_type == 'tabtransformer':
            return TabTransformerModel
        elif net_type == 'hashing':
            return HashingMLP
        elif net_type == 'ncf':
            return NCFModel
        else:
            raise ValueError(f'Unsupported net_type: {net_type}')
    
    def prepare_input_data(self):
        X_train_z_cols = [self.X_train[z_col] for z_col in self.X_train.columns[self.X_train.columns.str.startswith('z')]]
        X_test_z_cols = [self.X_test[z_col] for z_col in self.X_train.columns[self.X_train.columns.str.startswith('z')]]
        X_train_input = [self.X_train[self.exp_in.x_cols]] + X_train_z_cols
        X_test_input = [self.X_test[self.exp_in.x_cols]] + X_test_z_cols
        return X_train_input, X_test_input
    