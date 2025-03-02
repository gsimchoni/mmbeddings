import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Layer, Reshape, Hashing
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2

from mmbeddings.models.utils import build_coder


class EmbeddingsEncoder(Layer):
    def __init__(self, qs, d, l2reg_lambda, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embeddings = self.build_embeddings(qs, d, l2reg_lambda)
    
    def build_embeddings(self, qs, d, l2reg_lambda):
        embeddings = []
        for i, q in enumerate(qs):
            model = tf.keras.Sequential([
                Embedding(q, d, input_length=1, name='embed' + str(i),
                          embeddings_regularizer= None if l2reg_lambda is None else L2(l2reg_lambda)),
                Reshape(target_shape=(d,))
            ])
            embeddings.append(model)
        return embeddings
    
    def call(self, inputs):
        embeds = []
        for i, embedding in enumerate(self.embeddings):
            embeds.append(embedding(inputs[i]))
        return embeds


class HashingEncoder(Layer):
    def __init__(self, qs, hashing_bins, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embeddings = self.build_hashings(qs, hashing_bins)
    
    def build_hashings(self, qs, hashing_bins):
        hashings = []
        for i, q in enumerate(qs):
            hashing = Hashing(num_bins = hashing_bins, output_mode='multi_hot', sparse=True, name='hash' + str(i))
            hashings.append(hashing)
        return hashings
    
    def call(self, inputs):
        hashs = []
        for i, hashing in enumerate(self.embeddings):
            hashs.append(hashing(inputs[i]))
        return hashs


class EmbeddingsDecoder(Layer):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.nn = build_coder(self.input_dim, self.exp_in.n_neurons,
                              self.exp_in.dropout, self.exp_in.activation)
        self.concat = Concatenate()
        self.dense_output = Dense(1, activation=last_layer_activation)

    def call(self, X_input, embeds):
        concat = self.concat([X_input] + embeds)
        out_hidden = self.nn(concat)
        output = self.dense_output(out_hidden)
        return output


class EmbeddingsDecoderGrowthModel(Layer):
    """"""

    def __init__(self, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta_1 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True, name="beta_1")
        self.beta_2 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True, name="beta_2")
        self.beta_3 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True, name="beta_3")

    def call(self, X_input, embeds):
        Z_0, Z_1, Z_2 = embeds[0][:, 0:1], embeds[0][:, 1:2], embeds[0][:, 2:3]
        numerator = self.beta_1 + Z_0
        denominator = 1 + tf.math.exp(-(X_input - (self.beta_2 + Z_1)) / tf.maximum(self.beta_3 + Z_2, 1e-1))
        output = tf.math.divide_no_nan(numerator, denominator)
        return output


class EmbeddingsMLP(Model):
    def __init__(self, exp_in, input_dim, last_layer_activation, growth_model, l2reg_lambda, feature_hashing):
        """
        Multi-layer perceptron model with embeddings.
        """
        super(EmbeddingsMLP, self).__init__()
        self.exp_in = exp_in
        self.input_dim = input_dim
        # self.callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]
        if feature_hashing:
            self.encoder = HashingEncoder(self.exp_in.qs, self.exp_in.hashing_bins)
            decoder_input_dim = self.input_dim + self.exp_in.hashing_bins * len(self.exp_in.qs)
        else:
            self.encoder = EmbeddingsEncoder(self.exp_in.qs, self.exp_in.d, l2reg_lambda)
            decoder_input_dim = self.input_dim + self.exp_in.d * len(self.exp_in.qs)
        if growth_model:
            self.decoder = EmbeddingsDecoderGrowthModel()
        else:
            self.decoder = EmbeddingsDecoder(self.exp_in, decoder_input_dim, last_layer_activation)
            
    def call(self, inputs):
        """
        Build the MLP model with embeddings.
        """
        X_input = inputs[0]
        Z_inputs = inputs[1:]
        embeds = self.encoder(Z_inputs)
        output = self.decoder(X_input, embeds)
        return output
    
    def fit_model(self, X_train, y_train):
        history = self.fit(X_train, y_train,
                           epochs=self.exp_in.epochs, callbacks=self.callbacks,
                           batch_size=self.exp_in.batch, validation_split=0.1,
                           verbose=self.exp_in.verbose)
        return history
    
    def summarize(self, y_test, y_pred, history, sig2bs_hat_list):
        if self.exp_in.y_type == 'continuous':
            metric = np.mean((y_test - y_pred.reshape(-1)) ** 2)
        elif self.exp_in.y_type == 'binary':
            metric = roc_auc_score(y_test, y_pred)
        else:
            raise ValueError(f'Unsupported y_type: {self.exp_in.y_type}')
        sig2bs_mean_est = [np.mean(sig2bs) for sig2bs in sig2bs_hat_list]
        sigmas = [np.nan, sig2bs_mean_est]
        nll_tr, nll_te = np.nan, np.nan
        n_epochs = len(history.history['loss'])
        n_params = self.count_params()
        return metric, sigmas, nll_tr, nll_te, n_epochs, n_params