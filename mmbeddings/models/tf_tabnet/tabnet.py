import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Layer

from mmbeddings.models.base_model import BaseModel
from mmbeddings.models.embeddings import EmbeddingsEncoder
from mmbeddings.models.tf_tabnet.tabnet_model import TabNetEncoder
from mmbeddings.models.utils import build_coder


class TabNetDecoder(Layer):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.nn = build_coder(self.input_dim, self.exp_in.n_neurons_decoder,
                              self.exp_in.dropout, self.exp_in.activation)
        self.dense_output = Dense(1, activation=last_layer_activation)

    def call(self, inputs):
        out_hidden = self.nn(inputs)
        output = self.dense_output(out_hidden)
        return output


class TabNetModel(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Multi-layer perceptron model with embeddings.
        """
        super(TabNetModel, self).__init__(exp_in, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
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
