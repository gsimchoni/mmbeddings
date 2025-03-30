import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Layer, Reshape, Hashing
from tensorflow.keras.regularizers import L2

from mmbeddings.models.base_model import BaseModel
from mmbeddings.models.unifiedembeddings import UnifiedEmbedding
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


class UnifiedEmbeddingsEncoder(Layer):
    def __init__(self, q, d, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embeddings = UnifiedEmbedding(q, d, name='embed')
        self.reshape = Reshape(target_shape=(d,))
    
    def call(self, inputs):
        embeds = []
        for i, input in enumerate(inputs):
            embeds.append(self.reshape(self.embeddings(input, fnum=[i])))
        return embeds


class EmbeddingsDecoder(Layer):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.nn = build_coder(self.input_dim, self.exp_in.n_neurons_decoder,
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


class EmbeddingsMLP(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, growth_model, l2reg_lambda, **kwargs):
        """
        Multi-layer perceptron model with embeddings.
        """
        super(EmbeddingsMLP, self).__init__(exp_in, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
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


class HashingMLP(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Multi-layer perceptron model with embeddings.
        """
        super(HashingMLP, self).__init__(exp_in, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.encoder = HashingEncoder(self.exp_in.qs, self.exp_in.hashing_bins)
        decoder_input_dim = self.input_dim + self.exp_in.hashing_bins * len(self.exp_in.qs)
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


class UnifiedEmbeddingsMLP(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Multi-layer perceptron model with unified hash embeddings.
        """
        super(UnifiedEmbeddingsMLP, self).__init__(exp_in, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        if len(self.exp_in.qs) == 1:
            single_q = self.exp_in.qs[0]
        else:
            single_q = self.exp_in.ue_q  # Arbitrary large number for hashing
        self.encoder = UnifiedEmbeddingsEncoder(single_q, self.exp_in.d)
        decoder_input_dim = self.input_dim + self.exp_in.d * len(self.exp_in.qs)
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
