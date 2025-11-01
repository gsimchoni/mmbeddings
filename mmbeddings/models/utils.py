import tensorflow as tf
from tensorflow.keras.layers import Layer, Add, Dense, LayerNormalization, MultiHeadAttention
import tensorflow.keras.backend as K

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

class Sampling(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(K.shape(log_var)) * K.exp(log_var / 2) + mean

def compute_category_embedding(cat_feature, embeddings, num_tokens, return_batch=True):
    """
    Given:
      cat_feature: Integer tensor of shape (B, 1) representing the categorical feature.
      embeddings:  Latent embeddings tensor of shape (B, E) for each observation.
      num_tokens:  Total number of categories.
      
    Returns:
      final_embeddings: Tensor of shape (B, E), where each observation gets the averaged
                        embedding for its category.
    """
    # Flatten cat_feature from shape (B, 1) to (B,)
    cat_feature_flat = tf.reshape(cat_feature, [-1])
    
    # Sum the latent embeddings for each category.
    # summed_embeddings will be of shape (num_tokens, E)
    summed_embeddings = tf.math.unsorted_segment_sum(embeddings,
                                                     cat_feature_flat,
                                                     num_segments=num_tokens)
    
    # Count the number of occurrences for each category.
    counts = tf.math.unsorted_segment_sum(tf.ones_like(cat_feature_flat, dtype=tf.float32),
                                          cat_feature_flat,
                                          num_segments=num_tokens)
    
    # Compute the average embedding for each category.
    avg_embeddings = tf.math.divide_no_nan(summed_embeddings, tf.expand_dims(counts, axis=-1))

    # Apply shrinkage factor: n_j / (n_j + 1)
    shrinkage = tf.expand_dims(counts / (counts + 1.0), axis=-1)  # shape (num_tokens, 1)
    avg_embeddings = avg_embeddings * shrinkage
    
    # Now, for each observation, look up the average embedding for its category.
    if return_batch:
        avg_embeddings = tf.gather(avg_embeddings, cat_feature_flat)
    
    return avg_embeddings

def compute_category_embedding_bayesian(cat_feature, mean_embeddings, log_var_embeddings, num_tokens, return_batch=True):
    """
    Bayesian aggregation of posterior embeddings with zero-count handling.
    """
    cat_feature_flat = tf.reshape(cat_feature, [-1])

    # Convert log variances to variances and compute precisions
    var_embeddings = tf.exp(log_var_embeddings)
    precision_embeddings = 1.0 / var_embeddings  # (B, E)

    # Sum precisions and weighted means per category
    summed_precision = tf.math.unsorted_segment_sum(precision_embeddings, cat_feature_flat, num_segments=num_tokens)
    weighted_sum_means = tf.math.unsorted_segment_sum(mean_embeddings * precision_embeddings, cat_feature_flat, num_segments=num_tokens)

    # Identify zero-count categories to avoid division by zero
    zero_precision_mask = tf.equal(summed_precision, 0.0)

    # Replace zero precisions with ones (temporary fix to avoid division by zero)
    safe_summed_precision = tf.where(zero_precision_mask, tf.ones_like(summed_precision), summed_precision)

    # Compute combined variance and mean safely
    combined_var = 1.0 / safe_summed_precision
    combined_mean = combined_var * weighted_sum_means

    # Restore safe defaults for zero-count categories (mean=0, var=1 or prior variance)
    combined_var = tf.where(zero_precision_mask, tf.ones_like(combined_var), combined_var)
    combined_mean = tf.where(zero_precision_mask, tf.zeros_like(combined_mean), combined_mean)

    # Convert combined variance back to log variance
    combined_log_var = tf.math.log(combined_var + 1e-8)

    if return_batch:
        combined_mean = tf.gather(combined_mean, cat_feature_flat)
        combined_log_var = tf.gather(combined_log_var, cat_feature_flat)

    return combined_mean, combined_log_var

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