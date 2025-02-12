import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Dense, Dropout


class BaseModel:
    def __init__(self, exp_in):
        """
        Base class for models.
        
        Parameters:
        exp_in : any - Input parameters or configuration for the model/experiment.
        """
        self.exp_in = exp_in

    def fit(self, X_train, y_train):
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.exp_in.batch, epochs=self.exp_in.epochs,
                                 validation_split=0.1, callbacks=self.callbacks,
                                 verbose=self.exp_in.verbose)
        return history
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test, verbose=self.exp_in.verbose, batch_size=self.exp_in.batch).reshape(-1)
        return y_pred
    
    def summarize(self, y_test, y_pred, history):
        if self.exp_in.y_type == 'continuous':
            metric = np.mean((y_test - y_pred) ** 2)
        elif self.exp_in.y_type == 'binary':
            metric = roc_auc_score(y_test, y_pred)
        else:
            raise ValueError(f'Unsupported y_type: {self.exp_in.y_type}')
        sigmas = (np.nan, [np.nan for _ in range(self.exp_in.n_sig2bs)])
        nll_tr, nll_te = np.nan, np.nan
        n_epochs = len(history.history['loss'])
        n_params = self.model.count_params()
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