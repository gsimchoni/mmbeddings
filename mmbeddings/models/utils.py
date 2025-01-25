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