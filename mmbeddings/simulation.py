from itertools import product
import pandas as pd

from mmbeddings.data import DataSimulator, ExperimentInput
from mmbeddings.experiments import Embeddings, IgnoreOHE, Mmbeddings
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
        self.counter = Count().gen()
        self.n_iter = params.get('n_iter', 5)
        self.n_sig2bs = len(params['sig2b_list'])
        self.n_categorical = len(params['q_list'])
        self.d = params.get('d', 10)
        self.qs_names =  list(map(lambda x: 'q' + str(x), range(self.n_categorical)))
        self.sig2bs_names =  list(map(lambda x: 'sig2b' + str(x), range(self.n_sig2bs)))
        self.sig2bs_est_names =  list(map(lambda x: 'sig2b_est' + str(x), range(self.n_sig2bs)))
        self.test_size = params.get('test_size', 0.2)
        self.pred_unknown_clusters = params.get('pred_unknown_clusters', False)
        self.exp_types = params['exp_types']
        self.verbose = params.get('verbose', False)

    def run(self):
        """
        Run the full simulation.
        """
        # Create an empty results DataFrame
        self.res_df = self.create_res_df()

        for N in self.params['N_list']:
            for sig2e in self.params['sig2e_list']:
                for qs in product(*self.params['q_list']):
                    for sig2bs in product(*self.params['sig2b_list']):
                        self.logger.info(f'N: {N}, test: {self.test_size:.2f}, qs: {", ".join(map(str, qs))}, '
                                                f'sig2e: {sig2e}, '
                                                f'sig2bs_mean: {", ".join(map(str, sig2bs))}')
                        for k in range(self.n_iter):
                            self.logger.info(f'Iteration {k + 1}/{self.n_iter}')
                            simulator = DataSimulator(qs, sig2e, sig2bs, N, self.test_size, self.pred_unknown_clusters, self.params)
                            exp_data = simulator.generate_data()
                            self.exp_in = ExperimentInput(exp_data, N, self.test_size, self.pred_unknown_clusters, qs, self.d,
                                                     sig2e, sig2bs, k, self.n_sig2bs, self.params).get()
                            self.iterate_experiment_types()

    def create_res_df(self):
        """
        Create an empty simulation results DataFrame.

        Returns:
        pd.DataFrame - An empty DataFrame for storing simulation results.
        """
        res_df = pd.DataFrame(columns=['N', 'test_size', 'batch', 'pred_unknown', 'sig2e'] +\
                              self.sig2bs_names + self.qs_names +\
                                ['experiment', 'exp_type', 'mse', 'sig2e_est'] +\
                                    self.sig2bs_est_names +\
                                        ['nll_train', 'nll_test'] + ['n_epochs', 'time'])
        return res_df
    
    def get_experiment(self, exp_type):
        """
        Instantiate the experiment.
        exp_type : str - The type of experiment to run.
        """
        if exp_type == 'ignore':
            experiment = IgnoreOHE(self.exp_in, ignore_RE=True)
        elif exp_type == 'ohe':
            experiment = IgnoreOHE(self.exp_in, ignore_RE=False)
        elif exp_type == 'embeddings':
            experiment = Embeddings(self.exp_in)
        elif exp_type == 'mmbeddings':
            experiment = Mmbeddings(self.exp_in)
        else:
            raise NotImplementedError(f'{exp_type} experiment not implemented.')
        return experiment

    def iterate_experiment_types(self):
        """
        Iterate through experiment types and run them on the data.
        """
        for exp_type in self.exp_types:
            if self.verbose:
                self.logger.info(f'experiment: {exp_type}')
            experiment = self.get_experiment(exp_type)
            experiment.run()
            res_summary = experiment.summarize()
            self.res_df.loc[next(self.counter)] = res_summary
            self.logger.debug(f'  Finished {exp_type}.')
        self.res_df.to_csv(self.out_file, float_format='%.6g')
