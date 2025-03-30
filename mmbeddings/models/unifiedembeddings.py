import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding


class UnifiedEmbedding(Layer):
    def __init__(self, emb_levels, emb_dim, **kwargs):
        super(UnifiedEmbedding, self).__init__(**kwargs)
        self.emb_levels = emb_levels
        self.emb_dim = emb_dim
        self.embedding = Embedding(emb_levels, emb_dim)

    def call(self, x, fnum):
        # x: Tensor of strings or categorical values (shape: batch_size,)
        # fnum: list of integers (hash seeds per feature)
        embeddings = []
        for seed in fnum:
            # Convert inputs to string if not already (for stable hashing)
            x_str = tf.strings.as_string(x)
            # TensorFlow hashing (efficient and vectorized)
            indices = tf.strings.to_hash_bucket_fast(x_str, num_buckets=self.emb_levels, name=None)
            # Mix with seed to simulate feature-specific hashing (to replicate seed-like behavior)
            indices = tf.math.floormod(indices + seed, self.emb_levels)
            emb = self.embedding(indices)
            embeddings.append(emb)
        # Concatenate embeddings for all seeds (features)
        combined_emb = tf.concat(embeddings, axis=-1)
        return combined_emb
