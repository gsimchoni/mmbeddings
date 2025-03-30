import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dot, Embedding, Flatten, Layer, Reshape, Hashing, LayerNormalization
from tensorflow.keras.regularizers import L2

from mmbeddings.models.base_model import BaseModel
from mmbeddings.models.unifiedembeddings import UnifiedEmbedding
from mmbeddings.models.utils import TransformerBlock, build_coder


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


class UnifiedNCFDecoder(Layer):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        self.qs = self.exp_in.qs
        self.n_RE_inputs = len(self.exp_in.qs)
        self.n_neurons = self.exp_in.n_neurons_decoder
        self.dropout = self.exp_in.dropout
        self.nn = build_coder(input_dim, self.n_neurons, self.dropout, self.exp_in.activation)
        self.dot_layer = Dot(axes=1)
        self.concat = Concatenate()
        self.dense_output = Dense(1, activation=last_layer_activation)

    def call(self, X_input, embeds):
        user_vector = embeds[0]
        item_vector = embeds[1]
        dot_user_item = self.dot_layer([user_vector, item_vector])  # Shape: (batch_size, 1)
        features_embedding_concat = self.concat([X_input, dot_user_item])
        out_hidden = self.nn(features_embedding_concat)
        output = self.dense_output(out_hidden)
        return output


class UnifiedTTDecoder(Layer):
    """"""

    def __init__(self, exp_in, input_dim, last_layer_activation, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.exp_in = exp_in
        self.last_layer_activation = last_layer_activation
        self.d = exp_in.d
        self.qs = exp_in.qs
        self.n_neurons = self.exp_in.n_neurons_decoder
        self.dropout = self.exp_in.dropout
        self.cont_norm = LayerNormalization(epsilon=1e-6, name="cont_norm")
        self.num_categorical = len(exp_in.qs)
        self.num_continuous = len(exp_in.x_cols)
        self.column_embedding = Embedding(input_dim=self.num_categorical, output_dim=self.d, name="column_embedding")
        self.num_transformer_blocks = getattr(exp_in, "num_transformer_blocks", 2)
        self.num_heads = getattr(exp_in, "num_heads", 4)
        self.ff_dim = getattr(exp_in, "ff_dim", 64)
        self.dropout_rate = getattr(exp_in, "dropout_rate", 0.1)
        self.transformer_blocks = [
            TransformerBlock(head_size=self.d, num_heads=self.num_heads, ff_dim=self.ff_dim,
                             dropout=self.dropout_rate, name=f"trans_block_{i}")
            for i in range(self.num_transformer_blocks)
        ]
        self.cat_flatten = Flatten(name="cat_flatten")
        self.nn = build_coder(input_dim, self.n_neurons, self.dropout, self.exp_in.activation)
        self.dense_output = Dense(1, activation=self.last_layer_activation, name="output")

    def call(self, X_input, embeds, training=False):
        X_cont = self.cont_norm(X_input)
        cat_embeds = tf.stack(embeds, axis=1)
        col_indices = tf.range(start=0, limit=self.num_categorical, delta=1)  # shape: (num_categorical,)
        col_embeds = self.column_embedding(col_indices)  # shape: (num_categorical, d)
        cat_embeds = cat_embeds + col_embeds  # broadcasting along the batch dimension
        x_cat = cat_embeds
        for block in self.transformer_blocks:
            x_cat = block(x_cat, training=training)
        cat_out = self.cat_flatten(x_cat)  # shape: (batch, num_categorical * d)
        combined = tf.concat([X_cont, cat_out], axis=-1)
        out_hidden = self.nn(combined)
        output = self.dense_output(out_hidden)
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

class UnifiedEmbeddingsNCF(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Multi-layer perceptron model with unified hash embeddings.
        """
        super(UnifiedEmbeddingsNCF, self).__init__(exp_in, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        if len(self.exp_in.qs) == 1:
            single_q = self.exp_in.qs[0]
        else:
            single_q = self.exp_in.ue_q
        self.encoder = UnifiedEmbeddingsEncoder(single_q, self.exp_in.d)
        decoder_input_dim = self.input_dim + 1
        self.decoder = UnifiedNCFDecoder(self.exp_in, decoder_input_dim, last_layer_activation)
            
    def call(self, inputs):
        """
        Build the MLP model with embeddings.
        """
        X_input = inputs[0]
        Z_inputs = inputs[1:]
        embeds = self.encoder(Z_inputs)
        output = self.decoder(X_input, embeds)
        return output


class UnifiedEmbeddingsTT(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Multi-layer perceptron model with unified hash embeddings.
        """
        super(UnifiedEmbeddingsTT, self).__init__(exp_in, **kwargs)
        self.exp_in = exp_in
        self.input_dim = input_dim
        if len(self.exp_in.qs) == 1:
            single_q = self.exp_in.qs[0]
        else:
            single_q = self.exp_in.ue_q
        self.encoder = UnifiedEmbeddingsEncoder(single_q, self.exp_in.d)
        decoder_input_dim = self.input_dim + self.exp_in.d * len(self.exp_in.qs)
        self.decoder = UnifiedTTDecoder(self.exp_in, decoder_input_dim, last_layer_activation)
            
    def call(self, inputs):
        """
        Build the MLP model with embeddings.
        """
        X_input = inputs[0]
        Z_inputs = inputs[1:]
        embeds = self.encoder(Z_inputs)
        output = self.decoder(X_input, embeds)
        return output
