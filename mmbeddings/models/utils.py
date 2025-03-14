from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.layers import Layer
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
    
    # Now, for each observation, look up the average embedding for its category.
    if return_batch:
        avg_embeddings = tf.gather(avg_embeddings, cat_feature_flat)
    
    return avg_embeddings

def adjusted_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return max(auc, 1 - auc)  # Flip if necessary

def adjusted_log_loss(y_true, y_pred):
    flipped_pred = 1 - y_pred  # Flip probabilities
    loss = log_loss(y_true, y_pred)
    flipped_loss = log_loss(y_true, flipped_pred)
    return min(loss, flipped_loss)  # Take the best alignment

def adjust_accuracy(y_true, y_pred):
    flipped_pred = 1 - y_pred  # Flip probabilities
    acc = accuracy_score(y_true, y_pred > 0.5)
    flipped_acc = accuracy_score(y_true, flipped_pred > 0.5)
    return max(acc, flipped_acc)  # Take the best alignment

def evaluate_predictions(y_type, y_test, y_pred):
        if y_type == 'continuous':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = [mse, mae, r2]
        elif y_type == 'binary':
            auc = adjusted_auc(y_test, y_pred)
            logloss = adjusted_log_loss(y_test, y_pred)
            accuracy = adjust_accuracy(y_test, y_pred)
            metrics = [auc, logloss, accuracy]
        else:
            raise ValueError(f'Unsupported y_type: {y_type}')
        return metrics