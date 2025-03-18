from itertools import product
import pandas as pd

from mmbeddings.data import DataSimulator, ExperimentInput
from mmbeddings.experiments import LMMNN, Embeddings, PrecomputedEmbeddingExperiment, REbeddings, TrainableEmbeddingExperiment
from mmbeddings.utils import Count

class Simulation:
    def __init__(self, out_file, params, logger):
        """
        Parameters:
        out_file : str - Path to the output file or directory.
        params : dict - Parameters for the simulation.
        logger : logging.Logger - Logger for the simulation.
        """
        self.out_file = out_file
        self.params = params
        self.logger = logger
        self.n_iter = params.get('n_iter', 5)
        self.n_sig2bs = len(params['sig2b_list'])
        self.n_categorical = len(params['q_list'])
        self.d = params.get('d', 10)
        self.qs_names =  list(map(lambda x: 'q' + str(x), range(self.n_categorical)))
        self.sig2bs_names =  list(map(lambda x: 'sig2b' + str(x), range(self.n_sig2bs)))
        self.sig2bs_est_names =  list(map(lambda x: 'sig2b_est' + str(x), range(self.n_sig2bs)))
        self.n_test = params.get('n_test', 10000)
        self.pred_unknown_clusters = params.get('pred_unknown_clusters', False)
        self.exp_types = params['exp_types']
        self.verbose = params.get('verbose', False)
        self.results = []
        self.metric_names = self.get_metric_names(params['y_type'])
        self.dtype_dict = self.get_dtype_dict()

    def run(self):
        """
        Run the full simulation.
        """
        for n_train in self.params['n_train_list']:
            for sig2e in self.params['sig2e_list']:
                for qs in product(*self.params['q_list']):
                    for sig2bs in product(*self.params['sig2b_list']):
                        for k in range(self.n_iter):
                            self.logger.info(f'Iteration {k + 1}/{self.n_iter}')
                            self.logger.info(f'n_train: {n_train}, qs: {", ".join(map(str, qs))}, '
                                                     f'sig2e: {sig2e}, '
                                                     f'sig2bs_mean: {", ".join(map(str, sig2bs))}, ')
                            simulator = DataSimulator(qs, sig2e, sig2bs, n_train, self.n_test, self.pred_unknown_clusters, self.params)
                            exp_data = simulator.generate_data()
                            for beta_vae in self.params['beta_vae_list']:
                                for batch in self.params['batch_list']:
                                    self.params['beta_vae'] = beta_vae
                                    self.params['batch'] = batch
                                    self.logger.info(f'beta_vae: {beta_vae}, '
                                                     f'batch: {batch}')
                                    self.exp_in = ExperimentInput(exp_data, n_train, self.n_test, self.pred_unknown_clusters, qs, self.d,
                                                                    sig2e, sig2bs, k, self.n_sig2bs, self.params).get()
                                    self.iterate_experiment_types()                                

    def get_dtype_dict(self):
        dtype_dict = {
            'n_train': 'int64',
            'n_test': 'int64',
            'batch': 'int64',
            'pred_unknown': 'bool',
            'sig2e': 'float64',
            'beta': 'float64',
            'experiment': 'int64',
            'exp_type': 'object',
            'frobenius': 'float64',
            'spearman': 'float64',
            'nrmse': 'float64',
            'auc_embed': 'float64',
            'metric_pre_post': 'float64',
            'sig2e_est': 'float64',
            'nll_train': 'float64',
            'nll_test': 'float64',
            'n_epochs': 'int64',
            'time': 'float64',
            'n_params': 'int64',
            'encoder': 'object',
            'decoder': 'object',
            'patience': 'int64'
        }
        dtype_dict.update({k: 'float64' for k in self.sig2bs_names + self.sig2bs_est_names})
        dtype_dict.update({k: 'int64' for k in self.qs_names})
        dtype_dict.update({k: 'float64' for k in self.metric_names})
        return dtype_dict
    
    def get_experiment(self, exp_type):
        """
        Instantiate the experiment.
        exp_type : str - The type of experiment to run.
        """
        if exp_type in ['ignore', 'ohe', 'mean-encoding', 'pca-encoding']:
            experiment = PrecomputedEmbeddingExperiment(self.exp_in, exp_type)
        elif exp_type == 'embeddings':
            experiment = Embeddings(self.exp_in)
        elif exp_type == 'embeddings-l2':
            experiment = Embeddings(self.exp_in, l2reg_lambda=0.1)
        elif exp_type == 'embeddings_growth_model':
            experiment = Embeddings(self.exp_in, growth_model=True)
        elif exp_type == 'mmbeddings':
            experiment = REbeddings(self.exp_in, REbeddings_type='mmbeddings')
        elif exp_type == 'mmbeddings_growth_model':
            experiment = REbeddings(self.exp_in, REbeddings_type='mmbeddings', growth_model=True)
        elif exp_type == 'mmcf':
            experiment = REbeddings(self.exp_in, REbeddings_type='mmbeddings', cf=True)
        elif exp_type == 'regbeddings':
            experiment = REbeddings(self.exp_in, REbeddings_type='regbeddings')
        elif exp_type == 'regbeddings_growth_model':
            experiment = REbeddings(self.exp_in, REbeddings_type='regbeddings', growth_model=True)
        elif exp_type == 'lmmnn':
            experiment = LMMNN(self.exp_in)
        elif exp_type == 'mmbeddings-v2':
            experiment = REbeddings(self.exp_in, REbeddings_type='mmbeddings-v2')
        elif exp_type in ['tabnet', 'tabtransformer', 'hashing', 'ncf']:
            experiment = TrainableEmbeddingExperiment(self.exp_in, exp_type)
        else:
            raise NotImplementedError(f'{exp_type} experiment not implemented.')
        return experiment
    
    def summarize(self, exp_type):
        """Summarize the results of the experiment."""
        return {
            'n_train': self.exp_in.n_train,
            'n_test': self.exp_in.n_test,
            'batch': self.exp_in.batch,
            'pred_unknown': self.exp_in.pred_unknown,
            'sig2e': self.exp_in.sig2e,
            'beta': self.exp_in.beta_vae,
            **{name: val for name, val in zip(self.sig2bs_names, self.exp_in.sig2bs)},  # Automatically expands sig2bs
            **{name: val for name, val in zip(self.qs_names, self.exp_in.qs)},  # Expands qs
            'experiment': self.exp_in.k,
            'exp_type': exp_type,
            **{name: val for name, val in zip(self.metric_names, self.exp_res.metrics)},
            'frobenius': self.exp_res.frobenius,
            'spearman': self.exp_res.spearman,
            'nrmse': self.exp_res.nrmse,
            'auc_embed': self.exp_res.auc_embed,
            'metric_pre_post': self.exp_res.metric_pre_post,
            'sig2e_est': self.exp_res.sigmas[0],
            **{name: val for name, val in zip(self.sig2bs_est_names, self.exp_res.sigmas[1])},
            'nll_train': self.exp_res.nll_tr,
            'nll_test': self.exp_res.nll_te,
            'n_epochs': self.exp_res.n_epochs,
            'time': self.exp_res.time,
            'n_params': self.exp_res.n_params,
            'encoder': '[' + ', '.join(map(str, self.exp_in.n_neurons_encoder)) + ']',
            'decoder': '[' + ', '.join(map(str, self.exp_in.n_neurons)) + ']',
            'patience': self.exp_in.patience if self.exp_in.patience is not None else self.exp_in.epochs
        }

    def get_metric_names(self, y_type):
        if y_type == 'continuous':
            return ['mse', 'mae', 'r2']
        elif y_type == 'binary':
            return ['auc', 'logloss', 'accuracy']
        else:
            raise ValueError(f'Unsupported y_type: {y_type}')
    
    def iterate_experiment_types(self):
        """
        Iterate through experiment types and run them on the data.
        """
        for exp_type in self.exp_types:
            if self.verbose:
                self.logger.info(f'experiment: {exp_type}')
            experiment = self.get_experiment(exp_type)
            experiment.run()
            self.exp_res = experiment.exp_res
            self.results.append(self.summarize(exp_type))
            self.logger.debug(f'  Finished {exp_type}.')
        self.res_df = pd.DataFrame(self.results).astype(self.dtype_dict)
        self.res_df.to_csv(self.out_file, float_format='%.6g')
