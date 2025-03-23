import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dot, Concatenate
from mmbeddings.models.utils import build_coder
from tensorflow.keras.regularizers import l2
from mmbeddings.models.base_model import BaseModel

class NCFModel(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Parameters:
          exp_in: an object with attributes:
             - x_cols: list of continuous feature names.
             - qs: list of cardinalities for the categorical features.
             - d: latent dimension for categorical embeddings.
          input_dim: unused
          last_layer_activation: activation for the final output.
        """
        super(NCFModel, self).__init__(exp_in, **kwargs)
        self.exp_in = exp_in
        self.last_layer_activation = last_layer_activation

        self.num_users = self.exp_in.qs[0]
        self.num_items = self.exp_in.qs[1]
        self.embedding_size = exp_in.d
        self.user_embedding = Embedding(
            self.num_users,
            self.embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=l2(1e-6),
        )
        self.user_bias = Embedding(self.num_users, 1)
        self.item_embedding = Embedding(
            self.num_items,
            self.embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=l2(1e-6),
        )
        self.item_bias = Embedding(self.num_items, 1)
        self.dot_layer = Dot(axes=2)
        self.concat = Concatenate()
        self.nn = build_coder(input_dim + 1, self.exp_in.n_neurons_decoder, self.exp_in.dropout, self.exp_in.activation)
        self.dense_output = Dense(1, activation=self.last_layer_activation, name="output")
        
    def call(self, inputs):
        """
        Expects inputs as a list:
          - inputs[0]: Continuous features tensor, shape (batch, num_continuous)
          - inputs[1:]: Each is an integer–encoded categorical feature, shape (batch,) or (batch, 1)
        """
        # Unpack inputs.
        X_input = inputs[0]
        user_input = inputs[1]
        item_input = inputs[2]
        
        user_vector = self.user_embedding(user_input)
        user_bias = tf.squeeze(self.user_bias(user_input), axis=-1)
        item_vector = self.item_embedding(item_input)
        item_bias = tf.squeeze(self.item_bias(item_input), axis=-1)
        dot_user_item = self.dot_layer([user_vector, item_vector])  # Shape: (batch_size, 1, 1)
        dot_user_item = tf.squeeze(dot_user_item, axis=-1)
        cat_out = dot_user_item + user_bias + item_bias
        features_embedding_concat = self.concat([X_input, cat_out])
        out_hidden = self.nn(features_embedding_concat)
        output = self.dense_output(out_hidden)
        return output
    