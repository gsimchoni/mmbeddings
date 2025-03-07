import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Concatenate, Dense, Dot, Layer
from tensorflow.keras.models import Model
from packaging import version
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

from mmbeddings.models.utils import Sampling, build_coder, compute_category_embedding


class MmbeddingsEncoder(Model):
    """"""

    def __init__(self, exp_in, input_dim, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.n_RE_inputs = len(self.exp_in.qs)
        self.d = self.exp_in.d
        self.n_neurons = self.exp_in.n_neurons_encoder
        self.dropout = self.exp_in.dropout
        self.dense_encoder_layers = []
        self.nn = build_coder(input_dim + 1, self.n_neurons, self.dropout, self.exp_in.activation)
        # self.nn = build_coder(input_dim + 1, [100, 100], self.dropout, self.exp_in.activation)
        self.concat = Concatenate()
        self.dense_mean_layers = []
        self.dense_log_var_layers = []
        for _ in range(self.n_RE_inputs):
            self.dense_mean_layers.append(Dense(self.d))
            self.dense_log_var_layers.append(Dense(self.d))
        self.sampling = Sampling()
            
    def call(self, inputs):
        X_input = inputs[0]
        y_input = inputs[1]
        X_y_concat = self.concat([X_input] + [y_input])
        z1 = self.nn(X_y_concat)
        mmbeddings_mean_list = []
        mmbeddings_log_var_list = []
        mmbeddings_list = []
        for i in range(self.n_RE_inputs):
            mmbeddings_mean = self.dense_mean_layers[i](z1)
            mmbeddings_mean_list.append(mmbeddings_mean)
            mmbeddings_log_var = self.dense_log_var_layers[i](z1)
            mmbeddings_log_var_list.append(mmbeddings_log_var)
            mmbeddings = self.sampling([mmbeddings_mean, mmbeddings_log_var])
            mmbeddings_list.append(mmbeddings)
        return mmbeddings_mean_list, mmbeddings_log_var_list, mmbeddings_list

class MmbeddingsDecoder(Model):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        self.n_neurons = self.exp_in.n_neurons
        self.dropout = self.exp_in.dropout
        self.nn = build_coder(input_dim, self.n_neurons, self.dropout, self.exp_in.activation)
        # self.nn = build_coder(input_dim, [100, 50, 25, 12], [0.25, 0.25, 0.25], self.exp_in.activation)
        self.concat = Concatenate()
        self.dense_output = Dense(1, activation=last_layer_activation)

    def call(self, X_input, Z_inputs, mmbeddings_list):
        ZB_list = []
        for i in range(self.n_RE_inputs):
            ZB = compute_category_embedding(Z_inputs[i], mmbeddings_list[i], self.qs[i])
            ZB_list.append(ZB)
        features_embedding_concat = self.concat([X_input] + ZB_list)
        out_hidden = self.nn(features_embedding_concat)
        output = self.dense_output(out_hidden)
        return output
    
    def predict_with_custom_B(self, X_input, B_input):
        X_B_combined = np.concatenate([X_input] + B_input, axis=1)
        X_B_combined = tf.convert_to_tensor(X_B_combined, dtype=tf.float32)
        out_hidden = self.nn(X_B_combined)
        output = self.dense_output(out_hidden)
        return output.numpy()


class MmbeddingsCFDecoder(Model):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        self.n_neurons = self.exp_in.n_neurons
        self.dropout = self.exp_in.dropout
        self.nn = build_coder(input_dim, self.n_neurons, self.dropout, self.exp_in.activation)
        # self.nn = build_coder(input_dim, [100, 50, 25, 12], [0.25, 0.25, 0.25], self.exp_in.activation)
        self.dot_layer = Dot(axes=1)
        self.concat = Concatenate()
        self.dense_output = Dense(1, activation=last_layer_activation)

    def call(self, X_input, Z_inputs, mmbeddings_list):
        user_vector = compute_category_embedding(Z_inputs[0], mmbeddings_list[0], self.qs[0])
        item_vector = compute_category_embedding(Z_inputs[1], mmbeddings_list[1], self.qs[1])
        dot_user_item = self.dot_layer([user_vector, item_vector])  # Shape: (batch_size, 1)
        features_embedding_concat = self.concat([X_input, dot_user_item])
        out_hidden = self.nn(features_embedding_concat)
        output = self.dense_output(out_hidden)
        return output
    
    def predict_with_custom_B(self, X_input, embeddings):
        user_vector, item_vector = embeddings
        user_vector = tf.convert_to_tensor(user_vector, dtype=tf.float32)
        item_vector = tf.convert_to_tensor(item_vector, dtype=tf.float32)
        X_input = tf.convert_to_tensor(X_input, dtype=tf.float32)
        dot_user_item = self.dot_layer([user_vector, item_vector])
        features_embedding_concat = self.concat([X_input, dot_user_item])
        out_hidden = self.nn(features_embedding_concat)
        output = self.dense_output(out_hidden)
        return output.numpy()


class MmbeddingsDecoderGrowthModel(Layer):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        self.concat = Concatenate()
        self.beta_1 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True, name="beta_1")
        self.beta_2 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True, name="beta_2")
        self.beta_3 = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True, name="beta_3")

    def call(self, X_input, Z_inputs, mmbeddings_list):
        # TODO: Refactor this to use the compute_category_embedding function.
        Z_mats = []
        for i in range(self.n_RE_inputs):
            Z_input = Z_inputs[i]
            if version.parse(tf.__version__) >= version.parse('2.8'):
                Z = CategoryEncoding(num_tokens=self.qs[i], output_mode='one_hot')(Z_input)
            else:
                Z = CategoryEncoding(max_tokens=self.qs[i], output_mode='binary')(Z_input)
            Z_mats.append(Z)
        ZB_list = []
        for i in range(self.n_RE_inputs):
            Z = Z_mats[i]
            decoder_re_inputs = mmbeddings_list[i]
            B = tf.math.divide_no_nan(K.dot(K.transpose(Z), decoder_re_inputs), K.reshape(K.sum(Z, axis=0), (self.qs[i], 1)))
            ZB = K.dot(Z, B)
            ZB_list.append(ZB)
        Z_0, Z_1, Z_2 = ZB_list[0][:, 0:1], ZB_list[0][:, 1:2], ZB_list[0][:, 2:3]
        numerator = self.beta_1 + Z_0
        denominator = 1 + tf.math.exp(tf.clip_by_value(-tf.math.divide_no_nan(X_input - (self.beta_2 + Z_1), tf.maximum(self.beta_3 + Z_2, 1e-1)), -50.0, 50.0))
        output = tf.math.divide_no_nan(numerator, denominator)
        return output, Z_mats
    
    def predict_with_custom_B(self, X_input, B_input):
        X_input = tf.convert_to_tensor(X_input, dtype=tf.float32)
        ZB = tf.convert_to_tensor(B_input[0], dtype=tf.float32)
        Z_0, Z_1, Z_2 = ZB[:, 0:1], ZB[:, 1:2], ZB[:, 2:3]
        numerator = self.beta_1 + Z_0
        denominator = 1 + tf.math.exp(-tf.math.divide(X_input - (self.beta_2 + Z_1), tf.maximum(self.beta_3 + Z_2, 1e-1)))
        output = tf.math.divide_no_nan(numerator, denominator)
        return output.numpy()


class MmbeddingsVAE(Model):
    def __init__(self, exp_in, input_dim, last_layer_activation, growth_model=False, cf=False):
        """
        MLP-based VAE model for mmbeddings.
        """
        super(MmbeddingsVAE, self).__init__()
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.encoder = MmbeddingsEncoder(exp_in, input_dim)
        self.d = self.exp_in.d
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        decoder_input_dim = self.input_dim + self.d * self.n_RE_inputs
        if growth_model:
            decoder_class = MmbeddingsDecoderGrowthModel
        elif cf:
            decoder_class = MmbeddingsCFDecoder
            decoder_input_dim = self.input_dim + 1
        else:
            decoder_class = MmbeddingsDecoder
        self.decoder = decoder_class(exp_in, decoder_input_dim, last_layer_activation)
        self.re_log_sig2b_prior = tf.constant(np.log(self.exp_in.re_sig2b_prior, dtype=np.float32))
        self.beta = self.exp_in.beta_vae
        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]
        
    def call(self, inputs):
        """
        Build the VAE model with mmbeddings.
        """
        X_input = inputs[0]
        y_input = inputs[1]
        Z_inputs = inputs[2:]
        mmbeddings_mean_list, mmbeddings_log_var_list, mmbeddings_list = self.encoder((X_input, y_input))
        output = self.decoder(X_input, Z_inputs, mmbeddings_list)

        self.add_loss_and_metrics(y_input, output, Z_inputs, mmbeddings_mean_list, mmbeddings_log_var_list)
        return output

    def add_loss_and_metrics(self, y_input, y_pred, Z_inputs, re_codings_mean_list, re_codings_log_var_list):
        for i in range(self.n_RE_inputs):
            re_codings_mean = compute_category_embedding(Z_inputs[i], re_codings_mean_list[i], self.qs[i], return_batch=False)
            re_codings_log_var = compute_category_embedding(Z_inputs[i], re_codings_log_var_list[i], self.qs[i], return_batch=False)
            # re_codings_log_var = tf.math.divide_no_nan(K.dot(K.transpose(Z0), K.exp(re_codings_log_var)), K.reshape(K.sum(Z0, axis=0)**2, (self.qs[i], 1)))
            # re_codings_log_var = K.log(tf.where(tf.equal(re_codings_log_var, 0), tf.ones_like(re_codings_log_var), re_codings_log_var))
            re_kl_loss_i = -0.5 * K.sum(
                1 + re_codings_log_var - self.re_log_sig2b_prior -
                K.exp(re_codings_log_var - self.re_log_sig2b_prior) - K.square(re_codings_mean) * K.exp(-self.re_log_sig2b_prior),
                axis=-1)
            # re_kl_loss_i = K.sum(tf.math.divide_no_nan(re_kl_loss_i, K.sum(Z0, axis=0)))# / self.exp_in.batch
            # re_kl_loss_i = K.sum(tf.math.multiply(re_kl_loss_i, K.sum(Z0, axis=0))) / self.exp_in.batch
            re_kl_loss_i = K.sum(re_kl_loss_i) / self.exp_in.batch
            if i == 0:
                re_kl_loss = re_kl_loss_i
            else:
                re_kl_loss += re_kl_loss_i
        if self.exp_in.y_type == 'continuous':
            # log_lik = 0.5 * K.sum(K.square(y_input - y_pred)) / self.exp_in.batch
            log_lik = 0.5 * MeanSquaredError()(y_input, y_pred)
        elif self.exp_in.y_type == 'binary':
            log_lik = BinaryCrossentropy()(y_input, y_pred)
        else:
            raise ValueError(f'Unsupported y_type: {self.exp_in.y_type}')
        self.add_loss(self.beta * re_kl_loss)
        self.add_loss(log_lik)
        self.add_metric(log_lik, name='log_loss')
        self.add_metric(re_kl_loss, name='re_kl_loss')
    
    def fit_model(self, X_train, Z_train, y_train, shuffle=True):
        history = self.fit([X_train] + [y_train] + Z_train, y_train,
                           epochs=self.exp_in.epochs, callbacks=self.callbacks,
                           batch_size=self.exp_in.batch, validation_split=0.1,
                           verbose=self.exp_in.verbose, shuffle=shuffle)
        return history
    
    def predict_embeddings(self, X_train, Z_train, y_train):
        _, _, mmbeddings_list = self.encoder.predict((X_train, y_train), verbose=self.exp_in.verbose, batch_size=100000)
        mmbeddings_list_processed = self.extract_mmbeddings(Z_train, mmbeddings_list)
        sig2bs_hat_list = [mmbeddings_list_processed[i].var(axis=0) for i in range(len(mmbeddings_list_processed))]
        return mmbeddings_list_processed, sig2bs_hat_list
    
    def extract_mmbeddings(self, Z_train, mmbeddings_list):
        B_df_list = []
        for i in range(self.n_RE_inputs):
            B_df = pd.DataFrame(mmbeddings_list[i])
            B_df['z'] = Z_train[i].values
            B_df_grouped = B_df.groupby('z')[B_df.columns[:self.d]].mean()
            B_df_grouped = B_df_grouped.reindex(range(self.qs[i]), fill_value=0)
            B_df_list.append(B_df_grouped.values)
        return B_df_list
    
    def predict_model(self, X_test, Z_test, B_hat_list):
        B_hat_list_processed = self.replicate_Bs_to_predict(Z_test, B_hat_list)
        y_pred = self.decoder.predict_with_custom_B(X_test, B_hat_list_processed).reshape(-1)
        return y_pred    

    def replicate_Bs_to_predict(self, Z_test, B_hat_list):
        """
        Replicates rows of each B_hat matrix according to the indices in Z_test columns.
        """
        B_hat_list_processed = []
        for Z, B_hat in zip(Z_test, B_hat_list):
            B_hat_processed = B_hat[Z.values]
            B_hat_list_processed.append(B_hat_processed)
        return B_hat_list_processed

    def summarize(self, y_test, y_pred, sig2bs_hat_list, losses_tr, losses_te, history):
        if self.exp_in.y_type == 'continuous':
            metric = np.mean((y_test - y_pred) ** 2)
        elif self.exp_in.y_type == 'binary':
            metric = roc_auc_score(y_test, y_pred)
        else:
            raise ValueError(f'Unsupported y_type: {self.exp_in.y_type}')
        sig2bs_mean_est = [np.mean(sig2bs) for sig2bs in sig2bs_hat_list]
        sigmas = [np.nan, sig2bs_mean_est]
        nll_tr, nll_te = losses_tr[0], losses_te[0]
        n_epochs = len(history.history['loss'])
        n_params = self.count_params()
        return metric, sigmas, nll_tr, nll_te, n_epochs, n_params
    
    def evaluate_model(self, X, Z, y):
        total_loss, squared_loss, re_kl_loss = self.evaluate([X] + [y] + Z, verbose=self.exp_in.verbose, batch_size=self.exp_in.batch)
        return total_loss, squared_loss, re_kl_loss
    
class MmbeddingsDecoderPostTraining(Model):
    def __init__(self, exp_in, decoder, exp_type):
        super(MmbeddingsDecoderPostTraining, self).__init__()
        self.exp_in = exp_in
        self.decoder = decoder
        self.exp_type = exp_type
        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs_post_training if self.exp_in.patience_post_training is None else self.exp_in.patience_post_training)]
    
    def call(self, inputs):
        X_input = inputs[0]
        mmbeddings_list = inputs[1]
        Z_inputs = inputs[2:]
        output = self.decoder(X_input, Z_inputs, mmbeddings_list)
        return output
    
    def fit_model(self, X_train, Z_train, embeddings_list_processed, y_train):
        self.fit([X_train] + [embeddings_list_processed] + Z_train,
                 y_train, verbose=self.exp_in.verbose,
                 batch_size=self.exp_in.batch, epochs=self.exp_in.epochs_post_training,
                 callbacks=self.callbacks, validation_split=0.1)
        