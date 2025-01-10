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


def build_coder(input_dim, n_neurons, dropout, activation):
        model = tf.keras.Sequential()

        if not n_neurons:
            # If no neurons specified, return an identity network.
            model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
            return model

        # Add the first Dense layer with input shape.
        model.add(tf.keras.layers.Dense(n_neurons[0], activation=activation, input_shape=(input_dim,)))

        # Add subsequent Dense layers and dropout layers (if specified).
        for i in range(1, len(n_neurons)):
            if dropout and len(dropout) >= i:  # Add dropout if specified.
                model.add(tf.keras.layers.Dropout(dropout[i - 1]))
            model.add(tf.keras.layers.Dense(n_neurons[i], activation=activation))

        return model

class BaseModel:
    def __init__(self, exp_in):
        """
        Base class for models.
        
        Parameters:
        exp_in : any - Input parameters or configuration for the model/experiment.
        """
        self.exp_in = exp_in

    def fit(self, X_train, y_train):
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.exp_in.batch, epochs=self.exp_in.epochs,
                                 validation_split=0.1, callbacks=self.callbacks,
                                 verbose=self.exp_in.verbose)
        return history
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test, verbose=self.exp_in.verbose).reshape(-1)
        return y_pred
    
    def summarize(self, y_test, y_pred, history):
        mse = np.mean((y_test - y_pred) ** 2)
        sigmas = (None, [None for _ in range(self.exp_in.n_sig2bs)])
        nll_tr, nll_te = None, None
        n_epochs = len(history.history['loss'])
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

class Encoder(Model):
    """"""

    def __init__(self, exp_in, input_dim, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.n_RE_inputs = len(self.exp_in.qs)
        self.d = self.exp_in.d
        self.n_neurons = self.exp_in.n_neurons
        self.dropout = self.exp_in.dropout
        self.dense_encoder_layers = []
        self.nn = build_coder(input_dim + 1, self.n_neurons, self.dropout, self.exp_in.activation)
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

class Decoder(Layer):
    """"""

    def __init__(self, exp_in, input_dim, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        self.n_neurons = self.exp_in.n_neurons
        self.dropout = self.exp_in.dropout
        self.nn = build_coder(input_dim, self.n_neurons, self.dropout, self.exp_in.activation)
        self.concat = Concatenate()
        self.dense_output = Dense(1)

    def call(self, X_input, Z_inputs, mmbeddings_list):
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
        features_embedding_concat = self.concat([X_input] + ZB_list)
        out_hidden = self.nn(features_embedding_concat)
        output = self.dense_output(out_hidden)
        return output, Z_mats
    
    def predict_with_custom_B(self, X_input, B_input):
        X_B_combined = np.concatenate([X_input] + B_input, axis=1)
        X_B_combined = tf.convert_to_tensor(X_B_combined, dtype=tf.float32)
        out_hidden = self.nn(X_B_combined)
        output = self.dense_output(out_hidden)
        return output.numpy()


class VAEMmbed(Model):
    def __init__(self, exp_in, input_dim, growth_model=False):
        """
        MLP-based VAE model for mmbeddings.
        """
        super(VAEMmbed, self).__init__()
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.encoder = Encoder(exp_in, input_dim)
        if growth_model:
            decoder_class = DecoderGrowthModel
        else:
            decoder_class = Decoder
        self.d = self.exp_in.d
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        decoder_input_dim = self.input_dim + self.d * self.n_RE_inputs
        self.decoder = decoder_class(exp_in, decoder_input_dim)
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
        Z_inputs = [inputs[2]]
        mmbeddings_mean_list, mmbeddings_log_var_list, mmbeddings_list = self.encoder((X_input, y_input))
        output, Z_mats = self.decoder(X_input, Z_inputs, mmbeddings_list)

        self.add_loss_and_metrics(y_input, output, Z_mats, mmbeddings_mean_list, mmbeddings_log_var_list)
        return output

    def add_loss_and_metrics(self, y_input, y_pred, Z_mats, re_codings_mean_list, re_codings_log_var_list):
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
        self.add_loss(self.beta * re_kl_loss)
        squared_loss = MeanSquaredError()(y_input, y_pred)
        self.add_loss(squared_loss)
        self.add_metric(squared_loss, name='squared_loss')
        self.add_metric(re_kl_loss, name='re_kl_loss')
    
    def fit_model(self, X_train, Z_train, y_train):
        history = self.fit([X_train] + [y_train] + Z_train, y_train,
                           epochs=self.exp_in.epochs, callbacks=self.callbacks,
                           batch_size=self.exp_in.batch, validation_split=0.1,
                           verbose=self.exp_in.verbose)
        return history
    
    def predict_mmbeddings(self, X_train, Z_train, y_train):
        _, _, mmbeddings_list = self.encoder.predict((X_train, y_train))
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
        y_pred = self.decoder.predict_with_custom_B(X_test, B_hat_list_processed).reshape(-1)
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

    def summarize(self, y_test, y_pred, sig2bs_hat_list, losses_tr, losses_te, history):
        mse = np.mean((y_test - y_pred) ** 2)
        sig2bs_mean_est = [np.mean(sig2bs) for sig2bs in sig2bs_hat_list]
        sigmas = [None, sig2bs_mean_est]
        nll_tr, nll_te = losses_tr[0], losses_te[0]
        n_epochs = len(history.history['loss'])
        return mse, sigmas, nll_tr, nll_te, n_epochs
    
    def evaluate_model(self, X, Z, y):
        total_loss, squared_loss, re_kl_loss = self.evaluate([X] + [y] + Z, verbose=self.exp_in.verbose)
        return total_loss, squared_loss, re_kl_loss

class DecoderGrowthModel(Layer):
    """"""

    def __init__(self, exp_in, input_dim, name="decoder", **kwargs):
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
        denominator = 1 + tf.math.exp(-(X_input - (self.beta_2 + Z_1)) / tf.maximum(self.beta_3 + Z_2, 1e-1))
        output = tf.math.divide_no_nan(numerator, denominator)
        return output, Z_mats
    
    def predict_with_custom_B(self, X_input, B_input):
        X_input = tf.convert_to_tensor(X_input, dtype=tf.float32)
        ZB = tf.convert_to_tensor(B_input[0], dtype=tf.float32)
        Z_0, Z_1, Z_2 = ZB[:, 0:1], ZB[:, 1:2], ZB[:, 2:3]
        numerator = self.beta_1 + Z_0
        denominator = 1 + tf.math.exp(-(X_input - (self.beta_2 + Z_1)) / tf.maximum(self.beta_3 + Z_2, 1e-1))
        output = tf.math.divide_no_nan(numerator, denominator)
        return output.numpy()
