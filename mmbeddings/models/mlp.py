from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from mmbeddings.models.base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, exp_in, input_dim):
        """
        Multi-layer perceptron model.
        """
        super().__init__(exp_in)
        self.input_dim = input_dim
        
    def build(self):
        """
        Build the MLP model.
        """
        self.model = Sequential()
        self.add_layers_sequential(self.input_dim)
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')

        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]