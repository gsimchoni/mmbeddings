from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from mmbeddings.models.base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, exp_in, input_dim, loss, last_layer_activation):
        """
        Multi-layer perceptron model.
        """
        super().__init__(exp_in)
        self.input_dim = input_dim
        self.loss = loss
        self.last_layer_activation = last_layer_activation
        
    def build(self):
        """
        Build the MLP model.
        """
        self.model = Sequential()
        self.add_layers_sequential(self.input_dim)
        self.model.add(Dense(1, activation=self.last_layer_activation))

        self.model.compile(loss=self.loss, optimizer='adam')

        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]