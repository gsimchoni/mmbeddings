import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Layer
from tensorflow.keras.models import Model

from mmbeddings.models.embeddings import EmbeddingsEncoder
from mmbeddings.models.tf_tabnet.tabnet_model import TabNetEncoder
from mmbeddings.models.utils import build_coder


class TabNetDecoder(Layer):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.nn = build_coder(self.input_dim, self.exp_in.n_neurons,
                              self.exp_in.dropout, self.exp_in.activation)
        self.dense_output = Dense(1, activation=last_layer_activation)

    def call(self, inputs):
        out_hidden = self.nn(inputs)
        output = self.dense_output(out_hidden)
        return output


class TabNetModel(Model):
    def __init__(self, exp_in, input_dim, last_layer_activation):
        """
        Multi-layer perceptron model with embeddings.
        """
        super(TabNetModel, self).__init__()
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]
        self.categorical_encoder = EmbeddingsEncoder(self.exp_in.qs, self.exp_in.d, l2reg_lambda=None)
        tabnet_params = {
            "decision_dim": 16, 
            "attention_dim": 16, 
            "n_steps": 5, 
            "n_shared_glus": 2, 
            "n_dependent_glus": 2, 
            "relaxation_factor": 1.5, 
            "epsilon": 1e-15, 
            "virtual_batch_size": None, 
            "momentum": 0.98, 
            "mask_type": "entmax", 
            "lambda_sparse": 1e-4,
        }
        self.tabnet_encoder = TabNetEncoder(**tabnet_params)
        decoder_input_dim = tabnet_params['decision_dim']
        self.decoder = TabNetDecoder(self.exp_in, decoder_input_dim, last_layer_activation)
        self.concat = Concatenate()

    def call(self, inputs):
        """
        Build the TabNet.
        """
        X_input = inputs[0]
        Z_inputs = inputs[1:]
        embeds = self.categorical_encoder(Z_inputs)
        encoded_inputs = self.concat([X_input] + embeds)
        embeds = self.tabnet_encoder(encoded_inputs)
        output = self.decoder(embeds)
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