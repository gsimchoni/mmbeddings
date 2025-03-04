import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Add, Concatenate, Dense, Flatten, Layer, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model

from mmbeddings.models.embeddings import Embedding

class TransformerBlock(Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, **kwargs):
        """
        Transformer encoder block as in the Keras TabTransformer example.
        """
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.add1 = Add()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(head_size)
        ])
        self.add2 = Add()
        self.norm2 = LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=False):
        # Multi-head attention on inputs.
        attn_output = self.mha(inputs, inputs, training=training)
        # Residual connection + normalization.
        x = self.add1([inputs, attn_output])
        x = self.norm1(x)
        # Feed-forward network.
        ffn_output = self.ffn(x)
        # Residual connection + normalization.
        x = self.add2([x, ffn_output])
        x = self.norm2(x)
        return x

# --- TabTransformer Model Definition Following Keras Example ---
class TabTransformerModel(Model):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Parameters:
          exp_in: an object with attributes:
             - x_cols: list of continuous feature names.
             - qs: list of cardinalities for the categorical features.
             - d: latent dimension for categorical embeddings.
             - (optionally) num_transformer_blocks, num_heads, ff_dim, dropout.
          input_dim: unused
          last_layer_activation: activation for the final output.
        """
        super().__init__(**kwargs)
        self.exp_in = exp_in
        self.last_layer_activation = last_layer_activation
        self.d = exp_in.d  # latent dimension for embeddings
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]

        # Continuous branch: simply apply LayerNormalization.
        self.cont_norm = LayerNormalization(epsilon=1e-6, name="cont_norm")
        self.num_continuous = len(exp_in.x_cols)
        
        # Categorical branch:
        self.num_categorical = len(exp_in.qs)
        # Create one embedding layer per categorical feature.
        self.cat_embeddings = []
        for i, q in enumerate(exp_in.qs):
            emb_layer = Embedding(
                input_dim=q, output_dim=self.d, name=f"cat_emb_{i}"
            )
            self.cat_embeddings.append(emb_layer)
        # Create column embeddings (one per categorical column).
        self.column_embedding = Embedding(
            input_dim=self.num_categorical, output_dim=self.d, name="column_embedding"
        )
        
        # Transformer blocks for the categorical features.
        self.num_transformer_blocks = getattr(exp_in, "num_transformer_blocks", 2)
        self.num_heads = getattr(exp_in, "num_heads", 4)
        self.ff_dim = getattr(exp_in, "ff_dim", 64)
        self.dropout_rate = getattr(exp_in, "dropout_rate", 0.1)
        self.transformer_blocks = [
            TransformerBlock(head_size=self.d, num_heads=self.num_heads, ff_dim=self.ff_dim,
                             dropout=self.dropout_rate, name=f"trans_block_{i}")
            for i in range(self.num_transformer_blocks)
        ]
        # Instead of global average pooling, we flatten the output.
        self.cat_flatten = Flatten(name="cat_flatten")
        
        # Combine the two branches.
        self.combined_dense = Dense(64, activation="relu", name="combined_dense")
        self.out_dense = Dense(1, activation=self.last_layer_activation, name="output")
        
    def call(self, inputs, training=False):
        """
        Expects inputs as a list:
          - inputs[0]: Continuous features tensor, shape (batch, num_continuous)
          - inputs[1:]: Each is an integer–encoded categorical feature, shape (batch,) or (batch, 1)
        """
        # Unpack inputs.
        X_input = inputs[0]
        Z_inputs = inputs[1:]
        
        # Process continuous features: apply LayerNormalization.
        X_cont = self.cont_norm(X_input)  # shape: (batch, num_continuous)
        
        # Process categorical features.
        cat_embeds = []
        for i, z in enumerate(Z_inputs):
            # Ensure shape is (batch,) if given as (batch, 1)
            if len(z.shape) == 2 and z.shape[-1] == 1:
                z = tf.squeeze(z, axis=-1)
            emb = self.cat_embeddings[i](z)  # shape: (batch, d)
            cat_embeds.append(emb)
        # Stack embeddings into shape: (batch, num_categorical, d)
        cat_embeds = tf.stack(cat_embeds, axis=1)
        
        # Create and add column embeddings.
        col_indices = tf.range(start=0, limit=self.num_categorical, delta=1)  # shape: (num_categorical,)
        col_embeds = self.column_embedding(col_indices)  # shape: (num_categorical, d)
        cat_embeds = cat_embeds + col_embeds  # broadcasting along the batch dimension
        
        # Process through transformer blocks.
        x_cat = cat_embeds
        for block in self.transformer_blocks:
            x_cat = block(x_cat, training=training)
        
        # Instead of global average pooling, flatten the categorical branch.
        cat_out = self.cat_flatten(x_cat)  # shape: (batch, num_categorical * d)
        
        # Concatenate the continuous features with the flattened categorical features.
        combined = tf.concat([X_cont, cat_out], axis=-1)
        x = self.combined_dense(combined)
        output = self.out_dense(x)
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