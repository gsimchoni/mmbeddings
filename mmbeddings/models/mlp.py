from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from mmbeddings.models.base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, exp_in, input_dim, last_layer_activation, **kwargs):
        """
        Multi-layer perceptron model.
        """
        super(MLP, self).__init__(exp_in, **kwargs)
        self.input_dim = input_dim
        self.last_layer_activation = last_layer_activation
        self.model = Sequential()
        self.add_layers_sequential(self.input_dim)
        self.model.add(Dense(1, activation=self.last_layer_activation))
        
    def call(self, inputs):
        return self.model(inputs)
