from mmbeddings.utils import ExpData, ExpInput


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSimulator:
    def __init__(self, qs, sig2e, sig2bs, N, test_size, pred_unknown_clusters, params):
        """
        Parameters:
        qs : list[int] - Number of clusters for each random effect.
        sig2e : float - Variance of the Gaussian noise.
        sig2bs : list[float] - Variances for each random effect embedding.
        N : int - Total number of samples.
        test_size : float - Proportion of data to be used as test set.
        pred_unknown_clusters : bool - Whether to predict unknown clusters during testing.
        params : dict - Additional parameters for fixed effects.
        """
        self.qs = qs
        self.sig2e = sig2e
        self.sig2bs = sig2bs
        self.N = N
        self.test_size = test_size
        self.pred_unknown_clusters = pred_unknown_clusters
        self.params = params
        self.p = params.get('n_fixed_effects', 10)
        self.sig2bs_means = params.get('sig2bs_means', [1.0] * len(qs))
        self.sig2bs_identical = params.get('sig2bs_identical', False)
        self.d = params.get('d', 10)
        self.n_per_cat = params.get('n_per_cat', 30)

    def generate_data(self):
        """Generate the simulated dataset."""
        X = self.sample_fe()
        Z_idx_list, B_list = self.sample_re()
        noise = self.sample_noise()
        y = self.calculate_y(X, B_list, Z_idx_list, noise)
        df, x_cols = self.create_df(X, Z_idx_list, y)
        X_train, X_test, y_train, y_test = self.split_data(df)
        return ExpData(X_train, X_test, y_train, y_test, x_cols)

    def sample_fe(self):
        """Sample fixed effects."""
        X = np.random.uniform(-1, 1, self.N * self.p).reshape((self.N, self.p))
        return X

    def sample_re(self):
        """Sample random effects embeddings"""
        Z_idx_list = []
        B_list = []
        for k, q in enumerate(self.qs):
            sig2bs_mean = self.sig2bs_means[k]
            if sig2bs_mean < 1:
                fs_factor = sig2bs_mean
            else:
                fs_factor = 1
            if self.sig2bs_identical:
                sig2bs = np.repeat(sig2bs_mean, self.d)
            else:
                sig2bs = (np.random.poisson(sig2bs_mean, self.d) + 1) * fs_factor
            D = np.diag(sig2bs)
            B = np.random.multivariate_normal(np.zeros(self.d), D, q)
            B_list.append(B)
            fs = np.random.poisson(self.n_per_cat, q) + 1
            fs_sum = fs.sum()
            ps = fs / fs_sum
            ns = np.random.multinomial(self.N, ps)
            Z_idx = np.repeat(range(q), ns)
            # Z = self.get_dummies(Z_idx, q)
            # ZB_list.append(Z @ B)
            Z_idx_list.append(Z_idx)
        return Z_idx_list, B_list

    def sample_noise(self):
        """Sample Gaussian noise."""
        e = np.random.normal(0, np.sqrt(self.sig2e), self.N)
        return e

    def calculate_y(self, X, B_list, Z_idx_list, noise):
        """Generate a non-linear response vector y."""
        # Calculate embeddings for each categorical feature
        embeddings = []
        for k in range(len(self.qs)):
            B_k = B_list[k]  # q_k x d embedding matrix
            Z_idx_k = Z_idx_list[k]  # N-vector with levels for feature k
            embeddings.append(B_k[Z_idx_k])  # N x d embedding for feature k

        # Concatenate all embeddings along with the continuous features
        embeddings_concat = np.hstack(embeddings)  # N x (K * d)
        input_features = np.hstack([X, embeddings_concat])  # N x (p + K * d)

        # Define a non-linear function on the input features
        non_linear_term = np.sin(np.sum(input_features**2, axis=1, keepdims=True))

        # Ensure noise is reshaped to (N, 1) for consistency
        noise = noise.reshape(-1, 1) if noise.ndim == 1 else noise

        # Combine the non-linear term with noise
        y = non_linear_term + noise

        return y

    def create_df(self, X, Z_idx_list, y):
        """Create the final DataFrame."""
        df = pd.DataFrame(X)
        x_cols = ['X' + str(i) for i in range(X.shape[1])]
        df.columns = x_cols
        for k, Z_idx in enumerate(Z_idx_list):
            df['z' + str(k)] = Z_idx
        df['y'] = y
        return df, x_cols

    def split_data(self, df):
        """Split the DataFrame into train and test sets."""
        if self.pred_unknown_clusters:
            cluster_q = self.qs[0]
            train_clusters, test_clusters = train_test_split(range(cluster_q), test_size=self.test_size)
            X_train = df[df['z0'].isin(train_clusters)]
            X_test = df[df['z0'].isin(test_clusters)]
            y_train = df['y'][df['z0'].isin(train_clusters)]
            y_test = df['y'][df['z0'].isin(test_clusters)]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('y', axis=1), df['y'], test_size=self.test_size)
        return X_train, X_test, y_train, y_test

class ExperimentInput:
    def __init__(self, exp_data, N, test_size, pred_unknown_clusters, qs, d, sig2e, sig2bs, k, n_sig2bs, params):
        """
        Parameters:
        exp_data : namedtuple - Experiment data namedtuple.
        N : int - Total number of samples.
        test_size : float - Proportion of data to be used as test set.
        pred_unknown_clusters : bool - Whether to predict unknown clusters during testing.
        qs : list[int] - Number of clusters for each random effect.
        sig2e : float - Variance of the Gaussian noise.
        sig2bs : list[float] - Variances for each random effect embedding.
        k : int - Number of folds or splits for cross-validation.
        n_sig2bs : int - Number of random effect variances.
        params : dict - Additional parameters for the experiment.
        """
        self.exp_data = exp_data
        self.N = N
        self.test_size = test_size
        self.pred_unknown_clusters = pred_unknown_clusters
        self.qs = qs
        self.d = d
        self.sig2e = sig2e
        self.sig2bs = sig2bs
        self.k = k
        self.n_sig2bs = n_sig2bs
        self.params = params

    def get(self):
        """Return an ExpInput namedtuple."""
        return ExpInput(*self.exp_data, self.N, self.test_size, self.pred_unknown_clusters, self.qs, self.d, self.sig2e,
                        self.sig2bs, self.k, self.params['batch'], self.params['epochs'], self.params['patience'],
                        self.params['Z_embed_dim_pct'], self.n_sig2bs, self.params['verbose'], self.params['n_neurons'],
                        self.params['dropout'], self.params['activation'], self.params['RE_cols_prefix'],
                        self.params['re_sig2b_prior'], self.params['beta_vae'], self.params['log_params'])
