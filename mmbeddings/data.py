from mmbeddings.utils import ExpData, ExpInput


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSimulator:
    def __init__(self, qs, sig2e, sig2bs, n_train, n_test, pred_unknown_clusters, params):
        """
        Parameters:
        qs : list[int] - Number of clusters for each random effect.
        sig2e : float - Variance of the Gaussian noise.
        sig2bs : list[float] - Variances for each random effect embedding.
        n_train : int - Total number of training samples.
        n_test : int - Total number of testing samples.
        pred_unknown_clusters : bool - Whether to predict unknown clusters during testing.
        params : dict - Additional parameters for fixed effects.
        """
        self.qs = qs
        self.sig2e = sig2e
        self.sig2bs = sig2bs
        self.n_train = n_train
        self.n_test = n_test
        self.N = self.n_train + self.n_test
        self.test_size = self.n_test / self.N
        self.pred_unknown_clusters = pred_unknown_clusters
        self.params = params
        self.p = params.get('n_fixed_effects', 10)
        self.sig2bs_identical = params.get('sig2bs_identical', False)
        self.d = params.get('d', 10)
        self.n_per_cat = params.get('n_per_cat', 30)
        self.y_type = params.get('y_type', 'continuous')

    def generate_data(self):
        """Generate the simulated dataset."""
        X = self.sample_fe()
        # X = self.sample_fe_growth_model() # uncomment this line to sample fixed effects for growth model
        Z_idx_list, B_list = self.sample_re(X, X_dependent=False)
        noise = self.sample_noise()
        y = self.calculate_y(X, B_list, Z_idx_list, noise)
        df, x_cols = self.create_df(X, Z_idx_list, y)
        X_train, X_test, y_train, y_test = self.split_data(df)
        return ExpData(X_train, X_test, y_train, y_test, x_cols, B_list)

    def sample_fe(self):
        """Sample fixed effects."""
        X = np.random.uniform(-1, 1, self.N * self.p).reshape((self.N, self.p))
        return X
    
    def sample_fe_growth_model(self):
        """Sample fixed effects for growth model."""
        X = np.random.uniform(0, 20, self.N * self.p).reshape((self.N, self.p))
        return X

    def sample_re(self, X, X_dependent=False):
        """Sample random effects embeddings"""
        Z_idx_list = []
        B_list = []
        for k, q in enumerate(self.qs):
            sig2bs_mean = self.sig2bs[k]
            if sig2bs_mean < 1:
                fs_factor = sig2bs_mean
            else:
                fs_factor = 1
            if self.sig2bs_identical:
                sig2bs = np.repeat(sig2bs_mean, self.d)
            else:
                sig2bs = (np.random.poisson(sig2bs_mean, self.d) + 1) * fs_factor
            # sig2bs = [1.0, 1.0, 1.0] # uncomment this line to sample random effects for growth model
            D = np.diag(sig2bs)
            # D[0,1] = D[1,0] = 0.5 * np.sqrt(sig2bs[0] * sig2bs[1])
            # D[1,2] = D[2,1] = 0.5 * np.sqrt(sig2bs[1] * sig2bs[2])
            B = np.random.multivariate_normal(np.zeros(self.d), D, q)
            if X_dependent:
                try:
                    B2 = np.random.multivariate_normal(np.zeros(self.d), D, q)
                    # B += 0.1 * X[:q, :self.d]
                    B = np.where(X[:q, 0:1] > 0, B, B2)  # Use broadcasting to select rows
                except:
                    pass
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
        # embeddings_concat = np.hstack(embeddings)  # N x (K * d)
        # input_features = np.hstack([X, embeddings_concat])  # N x (p + K * d)

        # Define a non-linear function on the input features
        non_linear_term = non_linear_fn10(X, embeddings[0])
        # non_linear_term = growth_model(input_features)

        # Ensure noise is reshaped to (N, 1) for consistency
        noise = noise.reshape(-1, 1) if noise.ndim == 1 else noise

        if self.y_type == 'continuous':
            # Combine the non-linear term with noise
            y = non_linear_term + noise
            # y = np.maximum(0.01, y) # uncomment this line to add a lower bound to the response in a growth model
        elif self.y_type == 'binary':
            y = non_linear_term.flatten()
            p = np.exp(y - y.mean())/(1 + np.exp(y - y.mean()))
            y = np.random.binomial(1, p, size=len(y)).astype(float)
        else:
            raise ValueError(f'Unsupported y_type: {self.y_type}')

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
    def __init__(self, exp_data, n_train, n_test, pred_unknown_clusters, qs, d, sig2e, sig2bs, k, n_sig2bs, params):
        """
        Parameters:
        exp_data : namedtuple - Experiment data namedtuple.
        n_train : int - Total number of training samples.
        n_test : int - Total number of testing samples.
        pred_unknown_clusters : bool - Whether to predict unknown clusters during testing.
        qs : list[int] - Number of clusters for each random effect.
        sig2e : float - Variance of the Gaussian noise.
        sig2bs : list[float] - Variances for each random effect embedding.
        k : int - Number of folds or splits for cross-validation.
        n_sig2bs : int - Number of random effect variances.
        params : dict - Additional parameters for the experiment.
        """
        self.exp_data = exp_data
        self.n_train = n_train
        self.n_test = n_test
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
        return ExpInput(*self.exp_data, self.n_train, self.n_test, self.pred_unknown_clusters, self.qs, self.d, self.sig2e,
                        self.sig2bs, self.params['y_type'], self.k, self.params['batch'], self.params['epochs'], self.params['patience'],
                        self.params['Z_embed_dim_pct'], self.n_sig2bs, self.params['verbose'], self.params['n_neurons'],
                        self.params.get('n_neurons_encoder', self.params['n_neurons']),
                        self.params['dropout'], self.params['activation'], self.params['RE_cols_prefix'],
                        self.params['re_sig2b_prior'], self.params['beta_vae'], self.params['log_params'])

def non_linear_fn0(input_features):
    # Periodic sine function of the sum of squares of input features
    return np.sin(np.sum(input_features**2, axis=1, keepdims=True))

def non_linear_fn1(input_features):
    # Exponential weighted sum
    w = np.random.uniform(0.5, 1.5, size=input_features.shape[1])
    return np.exp(-np.sum((input_features * w)**2, axis=1, keepdims=True))

def non_linear_fn2(input_features):
    # Hyperbolic tangent of weighted interaction
    pairwise_products = np.triu(np.dot(input_features[:, :, None], input_features[:, None, :]), k=1)
    return np.tanh(pairwise_products.sum(axis=(1, 2), keepdims=True))

def non_linear_fn3(input_features):
    # ReLU followed by sine
    relu = np.maximum(0, np.sum(input_features, axis=1, keepdims=True))
    return np.sin(relu)

def non_linear_fn4(input_features):
    # Polynomial expansion with sigmoid
    poly = np.sum(input_features**3 - input_features, axis=1, keepdims=True)
    return 1 / (1 + np.exp(-poly))

def non_linear_fn5(input_features):
    # Gaussian RBF
    c = np.random.uniform(-1, 1, size=input_features.shape[1])
    return np.exp(-np.sum((input_features - c)**2, axis=1, keepdims=True))

def non_linear_fn6(input_features):
    # Periodic cosine interaction
    w = np.random.uniform(0.5, 1.5, size=input_features.shape[1])
    return np.cos(np.sum(input_features * w, axis=1, keepdims=True))

def non_linear_fn7(input_features):
    W = np.random.uniform(-1, 1, size=(input_features.shape[1], input_features.shape[1]))  # Weight matrix
    b = np.random.uniform(-0.5, 0.5, size=(input_features.shape[1],))  # Bias vector
    output = np.dot(input_features, W) + b
    return np.maximum(0, output[:, 0]).reshape(-1, 1)  # ReLU activation on the first neuron

def non_linear_fn8(X, embeddings_concat, d):
    betas = np.ones(d)
    Bbeta = embeddings_concat @ betas
    fB = Bbeta * np.cos(Bbeta) + 2 * embeddings_concat[:, 0] * embeddings_concat[:, 1]
    non_linear_term = non_linear_fn0(X) + fB.reshape(-1, 1)
    return non_linear_term

def non_linear_fn9(input_features):
    # Lindstrom and Bates (1990) Orange trees growth model
    x = input_features[:, 0]
    b1 = input_features[:, 1]
    b2 = input_features[:, 2]
    b3 = input_features[:, 3]
    beta = np.random.uniform(0.5, 1.5, size=3)
    beta_1, beta_2, beta_3 = beta
    non_linear_term = ((beta_1 + b1) * x) / (beta_2 + b2 + x) + (beta_3 + b3) * x
    non_linear_term = np.clip(non_linear_term, -1, 10)
    return non_linear_term

def growth_model(input_features):
    # Pinheiro and Bates (2000) Orange trees / Soybean growth model
    x = input_features[:, 0]
    b1 = input_features[:, 1]
    b2 = input_features[:, 2]
    b3 = input_features[:, 3]
    beta = [10.0, 5.0, 5.0]
    beta_1, beta_2, beta_3 = beta
    non_linear_term = (beta_1 + b1) / (1 + np.exp(-(x - (beta_2 + b2)) / (beta_3 + b3)))
    non_linear_term = non_linear_term.reshape(-1, 1)
    return non_linear_term

# Define the function f with random effects
def non_linear_fn10(X, B):
    """
    Compute the nonlinear function f with fixed and random effects using vectorized operations.

    Parameters:
    - X: n x 10 fixed effects matrix (numpy array)
    - B: n x 10 random effects matrix (numpy array)
    - beta_mean: length-10 mean vector for beta (fixed effects)
    - beta_sd: length-10 standard deviation vector for beta (random effects)

    Returns:
    - response: n x 1 response vector (numpy array)
    """
    n, p = X.shape
    assert p == 10, "X must have 10 columns for the fixed effects."
    assert B.shape == (n, 10), "B must be n x 10 for the random effects."

    # Sample a random beta vector (fixed effects)
    beta = np.array([1.0] * 10)

    # Add random effects to beta
    beta_with_b = beta + B

    # Compute the nonlinear function using vectorized operations
    term1 = 0.1 * np.exp(-beta_with_b[:, 0] * X[:, 0]**2)
    term2 = np.clip((beta_with_b[:, 1] * X[:, 1]) / (1 + beta_with_b[:, 2] * X[:, 2]**2), -5.0, 5.0)
    term3 = np.sin(beta_with_b[:, 3] * X[:, 3])
    term4 = beta_with_b[:, 4] * X[:, 4]
    term5 = (beta_with_b[:, 5] * X[:, 5]) / (1 + np.exp(-beta_with_b[:, 6] * X[:, 6]))
    term6 = np.arctan(beta_with_b[:, 7] * X[:, 7])
    term7 = beta_with_b[:, 8] * np.cos(X[:, 8])
    term8 = beta_with_b[:, 9] * np.log(1 + X[:, 9]**2)

    non_linear_term = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

    return non_linear_term.reshape(-1, 1)