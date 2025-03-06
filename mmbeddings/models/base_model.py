import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class BaseModel(Model):
    def __init__(self, exp_in):
        """
        Base class for models.
        
        Parameters:
        exp_in : any - Input parameters or configuration for the model/experiment.
        """
        super().__init__()
        self.exp_in = exp_in
        self.callbacks = [EarlyStopping(
            monitor='val_loss', patience=self.exp_in.epochs if self.exp_in.patience is None else self.exp_in.patience)]
    
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
    
    def add_layers_sequential(self, input_dim):
        n_neurons = self.exp_in.n_neurons
        dropout = self.exp_in.dropout
        activation = self.exp_in.activation
        if len(n_neurons) > 0:
            self.model.add(Dense(n_neurons[0], input_dim=input_dim, activation=activation))
            if dropout is not None and len(dropout) > 0:
                self.model.add(Dropout(dropout[0]))
            for i in range(1, len(n_neurons) - 1):
                self.model.add(Dense(n_neurons[i], activation=activation))
                if dropout is not None and len(dropout) > i:
                    self.model.add(Dropout(dropout[i]))
            if len(n_neurons) > 1:
                self.model.add(Dense(n_neurons[-1], activation=activation))
    
    def add_layers_functional(self, X_input, input_dim, n_neurons, dropout, activation):
        if len(n_neurons) > 0:
            x = Dense(n_neurons[0], input_dim=input_dim, activation=activation)(X_input)
            if dropout is not None and len(dropout) > 0:
                x = Dropout(dropout[0])(x)
            for i in range(1, len(n_neurons) - 1):
                x = Dense(n_neurons[i], activation=activation)(x)
                if dropout is not None and len(dropout) > i:
                    x = Dropout(dropout[i])(x)
            if len(n_neurons) > 1:
                x = Dense(n_neurons[-1], activation=activation)(x)
            return x
        return X_input