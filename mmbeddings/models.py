import numpy as np
from packaging import version
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Input, Reshape, Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.losses import MeanSquaredError

from mmbeddings.utils import get_dummies

class BaseModel:
    def __init__(self, exp_in):
        """
        Base class for models.
        
        Parameters:
        exp_in : any - Input parameters or configuration for the model/experiment.
        """
        self.exp_in = exp_in

    def fit(self, X_train, y_train):
        self.history = self.model.fit(X_train, y_train,
                                      batch_size=self.exp_in.batch, epochs=self.exp_in.epochs,
                                      validation_split=0.1, callbacks=self.callbacks,
                                      verbose=self.exp_in.verbose)
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test, verbose=self.exp_in.verbose).reshape(-1)
        return y_pred
    
    def summarize(self, y_test, y_pred):
        mse = np.mean((y_test - y_pred) ** 2)
        sigmas = (None, [None for _ in range(self.exp_in.n_sig2bs)])
        nll_tr, nll_te = None, None
        n_epochs = len(self.history.history['loss'])
        return mse, sigmas, nll_tr, nll_te, n_epochs
    
    def add_layers_sequential(self, input_dim):
        n_neurons = self.exp_in.n_neurons
        dropout = self.exp_in.dropout
        activation = self.exp_in.activation
        if len(n_neurons) > 0:
            self.model.add(Dense(n_neurons[0], input_dim=input_dim, activation=activation))
            if dropout is not None and len(dropout) > 0:
                self.model.add(Dropout(dropout[0]))
            for i in range(1, len(n_neurons) - 1):
                self.model.add(Dense(n_neurons[i], activation=activation))
                if dropout is not None and len(dropout) > i:
                    self.model.add(Dropout(dropout[i]))
            if len(n_neurons) > 1:
                self.model.add(Dense(n_neurons[-1], activation=activation))
    
    def add_layers_functional(self, X_input, input_dim, n_neurons, dropout, activation):
        n_neurons = self.exp_in.n_neurons
        dropout = self.exp_in.dropout
        activation = self.exp_in.activation
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


class MLP(BaseModel):
    def __init__(self, exp_in, input_dim):
        """
        Multi-layer perceptron model.
        """
        super().__init__(exp_in)
        self.input_dim = input_dim
        
    def build(self):
        """
        Build the MLP model.
        """
        self.model = Sequential()
        self.add_layers_sequential(self.input_dim)
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')

        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]


class MLPEmbed(BaseModel):
    def __init__(self, exp_in, input_dim):
        """
        Multi-layer perceptron model with embeddings.
        """
        super().__init__(exp_in)
        self.input_dim = input_dim
        
    def build(self):
        """
        Build the MLP model with embeddings.
        """
        embed_dim = self.exp_in.d
        X_input = Input(shape=(self.input_dim,))
        Z_inputs = []
        embeds = []
        qs_list = list(self.exp_in.qs)
        for i, q in enumerate(qs_list):
            Z_input = Input(shape=(1,))
            embed = Embedding(q, embed_dim, input_length=1, name='embed' + str(i))(Z_input)
            embed = Reshape(target_shape=(embed_dim,))(embed)
            Z_inputs.append(Z_input)
            embeds.append(embed)
        concat = Concatenate()([X_input] + embeds)
        concat_input_dim =  self.input_dim + embed_dim * len(qs_list)
        out_hidden = self.add_layers_functional(concat, concat_input_dim, self.exp_in.n_neurons, self.exp_in.dropout, self.exp_in.activation)
        output = Dense(1)(out_hidden)
        self.model = Model(inputs=[X_input] + Z_inputs, outputs=output)

        self.model.compile(loss='mse', optimizer='adam')

        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]

class Sampling(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(K.shape(log_var)) * K.exp(log_var / 2) + mean

class VAEMmbed(BaseModel):
    def __init__(self, exp_in, input_dim):
        """
        MLP-based VAE model for mmbeddings.
        """
        super().__init__(exp_in)
        self.input_dim = input_dim
        self.d = self.exp_in.d
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        self.re_log_sig2b_prior = tf.constant(np.log(self.exp_in.re_sig2b_prior, dtype=np.float32))
        self.beta = self.exp_in.beta_vae
        
    def build(self):
        """
        Build the VAE model with mmbeddings.
        """
        X_input = Input(shape=(self.input_dim,))
        y_input = Input(shape=(1,))
        X_y_concat = Concatenate()([X_input] + [y_input])
        z1 = self.add_layers_functional(X_y_concat, self.input_dim + 1)
        Z_inputs = []
        Z_mats = []
        for i in range(self.n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
            if version.parse(tf.__version__) >= version.parse('2.8'):
                Z = CategoryEncoding(num_tokens=self.qs[i], output_mode='one_hot')(Z_input)
            else:
                Z = CategoryEncoding(max_tokens=self.qs[i], output_mode='binary')(Z_input)
            Z_mats.append(Z)
        re_codings_mean_list = []
        re_codings_log_var_list = []
        re_codings_list = []
        for _ in range(self.n_RE_inputs):
            re_codings_mean = Dense(self.d)(z1)
            re_codings_mean_list.append(re_codings_mean)
            re_codings_log_var = Dense(self.d)(z1)
            re_codings_log_var_list.append(re_codings_log_var)
            re_codings = Sampling()([re_codings_mean, re_codings_log_var])
            re_codings_list.append(re_codings)
        self.variational_encoder = Model(
                inputs=[X_input] + [y_input], outputs=[re_codings_list])
        decoder_re_inputs_list = []
        for _ in range(self.n_RE_inputs):
            decoder_re_inputs = Input(shape=self.d)
            decoder_re_inputs_list.append(decoder_re_inputs)
        ZB_list = []
        for i in range(self.n_RE_inputs):
            Z = Z_mats[i]
            decoder_re_inputs = decoder_re_inputs_list[i]
            B = tf.math.divide_no_nan(K.dot(K.transpose(Z), decoder_re_inputs), K.reshape(K.sum(Z, axis=0), (self.qs[i], 1)))
            ZB = K.dot(Z, B)
            ZB_list.append(ZB)
        decoder_input_dim = self.input_dim + self.d * self.n_RE_inputs
        features_embedding_concat = Concatenate()([X_input] + ZB_list)
        out_hidden = self.add_layers_functional(features_embedding_concat, decoder_input_dim, self.exp_in.n_neurons, self.exp_in.dropout, self.exp_in.activation)
        output = Dense(1)(out_hidden)
        self.variational_decoder = Model(
                inputs=[X_input] + decoder_re_inputs_list + Z_inputs, outputs=[output])
        self.variational_decoder_no_Z = Model(inputs=[X_input] + ZB_list, outputs=[output])
        mmbeddings = self.variational_encoder([X_input] + [y_input])
        predictions = self.variational_decoder([X_input] + mmbeddings + Z_inputs)
        self.model = Model(inputs=[X_input] + [y_input] + Z_inputs, outputs=predictions)

        self.add_loss_and_metrics(y_input, predictions, Z_mats, re_codings_mean_list, re_codings_log_var_list)

        self.model.compile(optimizer='adam')

        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]

    def add_loss_and_metrics(self, y_input, predictions, Z_mats, re_codings_mean_list, re_codings_log_var_list):
        for i in range(self.n_RE_inputs):
            Z0 = Z_mats[i]
            re_codings_mean = re_codings_mean_list[i]
            re_codings_log_var = re_codings_log_var_list[i]
            re_codings_mean = tf.math.divide_no_nan(K.dot(K.transpose(Z0), re_codings_mean), K.reshape(K.sum(Z0, axis=0), (self.qs[i], 1)))            
            re_codings_log_var = tf.math.divide_no_nan(K.dot(K.transpose(Z0), K.exp(re_codings_log_var)), K.reshape(K.sum(Z0, axis=0)**2, (self.qs[i], 1)))
            re_codings_log_var = K.log(tf.where(tf.equal(re_codings_log_var, 0), tf.ones_like(re_codings_log_var), re_codings_log_var))
            re_kl_loss_i = -0.5 * K.sum(
                1 + re_codings_log_var - self.re_log_sig2b_prior -
                K.exp(re_codings_log_var - self.re_log_sig2b_prior) - K.square(re_codings_mean) * K.exp(-self.re_log_sig2b_prior),
                axis=-1)
            re_kl_loss_i = K.sum(re_kl_loss_i) / self.exp_in.batch
            if i == 0:
                re_kl_loss = re_kl_loss_i
            else:
                re_kl_loss += re_kl_loss_i
        self.model.add_loss(self.beta * re_kl_loss)
        squared_loss = MeanSquaredError()(y_input, predictions)
        self.model.add_loss(squared_loss)
        self.model.add_metric(squared_loss, name='squared_loss')
        self.model.add_metric(re_kl_loss, name='re_kl_loss')
    
    def fit(self, X_train, Z_train, y_train):
        self.history = self.model.fit([X_train] + [y_train] + Z_train, y_train,
                                      epochs=self.exp_in.epochs, callbacks=self.callbacks,
                                      batch_size=self.exp_in.batch, validation_split=0.1,
                                      verbose=self.exp_in.verbose)
    
    def predict_mmbeddings(self, X_train, Z_train, y_train):
        mmbeddings_list = self.variational_encoder.predict([X_train] + [y_train], verbose=0)
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
            B_df_list.append(B_df_grouped)
        return B_df_list
    
    def predict(self, X_test, Z_test, B_hat_list):
        B_hat_list_processed = self.replicate_Bs_to_predict(Z_test, B_hat_list)
        y_pred = self.variational_decoder_no_Z.predict(
            [X_test] + B_hat_list_processed, verbose=self.exp_in.verbose).reshape(-1)
        return y_pred    

    def replicate_Bs_to_predict(self, Z_test, B_hat_list):
        """
        Replicates rows of each B_hat matrix according to the indices in Z_test columns.
        """
        B_hat_list_processed = []
        for Z, B_hat in zip(Z_test, B_hat_list):
            B_hat_processed = B_hat.loc[Z.values]
            B_hat_list_processed.append(B_hat_processed)
        return B_hat_list_processed

    def summarize(self, y_test, y_pred, sig2bs_hat_list, losses_tr, losses_te):
        mse = np.mean((y_test - y_pred) ** 2)
        sig2bs_mean_est = [np.mean(sig2bs) for sig2bs in sig2bs_hat_list]
        sigmas = [None, sig2bs_mean_est]
        nll_tr, nll_te = losses_tr[0], losses_te[0]
        n_epochs = len(self.history.history['loss'])
        return mse, sigmas, nll_tr, nll_te, n_epochs
    
    def evaluate(self, X, Z, y):
        total_loss, squared_loss, re_kl_loss = self.model.evaluate([X] + [y] + Z, verbose=self.exp_in.verbose)
        return total_loss, squared_loss, re_kl_loss