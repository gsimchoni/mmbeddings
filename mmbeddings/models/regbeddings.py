import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Reshape
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.models import Model

from mmbeddings.models.embeddings import EmbeddingsDecoder, EmbeddingsDecoderGrowthModel
from mmbeddings.models.utils import Sampling, evaluate_predictions


class RegbeddingsEncoder(Model):
    """"""

    def __init__(self, qs, d, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embeddings = self.build_embeddings(qs, d)
        self.sampling = Sampling()
    
    def build_embeddings(self, qs, d):
        embeddings = []
        for i, q in enumerate(qs):
            model_mean = tf.keras.Sequential([
                Embedding(q, d, input_length=1),
                Reshape(target_shape=(d,))
            ], name='embed_mean' + str(i))
            model_log_var = tf.keras.Sequential([
                Embedding(q, d, input_length=1),
                Reshape(target_shape=(d,))
            ], name='embed_log_var' + str(i))
            embeddings.append((model_mean, model_log_var))
        return embeddings
    
    def call(self, inputs):
        embeds = []
        for i, (embed_mean, embed_log_var) in enumerate(self.embeddings):
            embeds.append(embed_mean(inputs[i]))
            embeds.append(embed_log_var(inputs[i]))
        regbeddings_list = []
        regbeddings_mean_list = []
        regbeddings_log_var_list = []
        for i, (embed_mean, embed_log_var) in enumerate(self.embeddings):
            regbeddings_mean = embed_mean(inputs[i])
            regbeddings_mean_list.append(regbeddings_mean)
            regbeddings_log_var = embed_log_var(inputs[i])
            regbeddings_log_var_list.append(regbeddings_log_var)
            regbeddings = self.sampling([regbeddings_mean, regbeddings_log_var])
            regbeddings_list.append(regbeddings)
        return regbeddings_mean_list, regbeddings_log_var_list, regbeddings_list


class RegbeddingsMLP(Model):
    def __init__(self, exp_in, input_dim, last_layer_activation, growth_model=False, cf=False):
        """
        MLP-based regularized embeddings model (Richman, 2024).
        """
        super(RegbeddingsMLP, self).__init__()
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.d = self.exp_in.d
        self.qs = self.exp_in.qs
        self.encoder = RegbeddingsEncoder(self.qs, self.d)
        self.n_RE_inputs = len(self.exp_in.qs)
        decoder_input_dim = self.input_dim + self.d * self.n_RE_inputs
        if growth_model:
            self.decoder = EmbeddingsDecoderGrowthModel()
        else:
            self.decoder = EmbeddingsDecoder(exp_in, decoder_input_dim, last_layer_activation)
        self.re_log_sig2b_prior = tf.constant(np.log(self.exp_in.re_sig2b_prior, dtype=np.float32))
        self.beta = self.exp_in.beta_vae
        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]
        
    def call(self, inputs):
        """
        Build the regularized embedding model with regbeddings.
        """
        X_input = inputs[0]
        y_input = inputs[1]
        Z_inputs = inputs[2:]
        regbeddings_mean_list, regbeddings_log_var_list, regbeddings_list = self.encoder(Z_inputs)
        output = self.decoder(X_input, regbeddings_list)

        self.add_loss_and_metrics(y_input, output, regbeddings_mean_list, regbeddings_log_var_list, self.exp_in.y_type)
        return output

    def add_loss_and_metrics(self, y_true, y_pred, re_codings_mean_list, re_codings_log_var_list, y_type):
        for i in range(self.n_RE_inputs):
            re_codings_mean = re_codings_mean_list[i]
            re_codings_log_var = re_codings_log_var_list[i]
            re_kl_loss_i = -0.5 * K.sum(
                1 + re_codings_log_var - self.re_log_sig2b_prior -
                K.exp(re_codings_log_var - self.re_log_sig2b_prior) - K.square(re_codings_mean) * K.exp(-self.re_log_sig2b_prior),
                axis=-1)
            re_kl_loss_i = K.mean(re_kl_loss_i)
            if i == 0:
                re_kl_loss = re_kl_loss_i
            else:
                re_kl_loss += re_kl_loss_i
        self.add_loss(self.beta * re_kl_loss)
        if y_type == 'continuous':
            log_lik = MeanSquaredError()(y_true, y_pred)
        elif y_type == 'binary':
            log_lik = BinaryCrossentropy()(y_true, y_pred)
        else:
            raise ValueError(f'Unsupported y_type: {y_type}')
        self.add_loss(log_lik)
        self.add_metric(log_lik, name='squared_loss')
        self.add_metric(re_kl_loss, name='re_kl_loss')
    
    def fit_model(self, X_train, Z_train, y_train, shuffle=True):
        history = self.fit([X_train] + [y_train] + Z_train, y_train,
                           epochs=self.exp_in.epochs, callbacks=self.callbacks,
                           batch_size=self.exp_in.batch, validation_split=0.1,
                           verbose=self.exp_in.verbose, shuffle=shuffle)
        return history
    
    def predict_embeddings(self, X_train, Z_train, y_train):
        regbeddings_list = []
        for i in range(self.n_RE_inputs):
            regbeddings = self.encoder.get_layer('embed_mean' + str(i)).get_weights()[0]
            regbeddings_list.append(regbeddings)
        sig2bs_hat_list = [regbeddings_list[i].var(axis=0) for i in range(len(regbeddings_list))]
        return regbeddings_list, sig2bs_hat_list
    
    def predict_model(self, X_test, Z_test, embeddings_list):
        dummy_y_test = np.random.normal(size=(X_test.shape[0],))
        y_pred = self.predict([X_test] + [dummy_y_test] + Z_test, verbose=self.exp_in.verbose, batch_size=self.exp_in.batch).reshape(-1)
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
        metrics = evaluate_predictions(self.exp_in.y_type, y_test, y_pred)
        sig2bs_mean_est = [np.mean(sig2bs) for sig2bs in sig2bs_hat_list]
        sigmas = [np.nan, sig2bs_mean_est]
        nll_tr, nll_te = losses_tr[0], losses_te[0]
        n_epochs = len(history.history['loss'])
        n_params = self.count_params()
        return metrics, sigmas, nll_tr, nll_te, n_epochs, n_params
    
    def evaluate_model(self, X, Z, y):
        total_loss, squared_loss, re_kl_loss = self.evaluate([X] + [y] + Z, verbose=self.exp_in.verbose, batch_size=self.exp_in.batch)
        return total_loss, squared_loss, re_kl_loss