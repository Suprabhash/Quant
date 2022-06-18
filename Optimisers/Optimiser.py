import itertools
import numpy as np
import psutil
from Optimisers.MCMC.MCMC import MCMC
import math
import psutil
from pathos.multiprocessing import ProcessingPool

def brute_force_optimizer(*params, alpha, optim, parallelize):
    def  brute_force(inp):
        return optim(alpha(inp))
    params = params[0]
    inputs = list(itertools.product(*params))
    if parallelize:
        pool = ProcessingPool(nodes=7)
        results = pool.map(brute_force, inputs)
        pool.clear()
    else:
        results = [optim(alpha(inputs[i])) for i in range(len(inputs))]
    return results


class Optimiser():

    def __init__(self, method):
        self.method = method
        self.data = None
        self.metrics_searchspace = None

        def optim_function(a):
            return a

        self.optim_function = optim_function

    def define_alpha_function(self, alpha_function):
        self.alpha_function = alpha_function

    def define_optim_function(self, optim_function):
        if optim_function == None:
            self.optim_function = None
        else:
            self.optim_function = (lambda x, y, bins: optim_function(x))

    def define_prior(self, prior):
        self.prior = prior

    def define_guess(self, guess):
        self.guess = guess

    def define_target(self, target):
        self.target = target

    def define_lower_and_upper_limit(self, ll, ul):
        self.lower_limit = ll
        self.upper_limit = ul

    def define_iterations(self, iters):
        self.iters = iters

    def define_parameter_searchspace(self, parameter_searchspace):
        self.parameter_searchspace = parameter_searchspace

    def define_metrics_searchspace(self, metrics_searchspace):
        self.metrics_searchspace = metrics_searchspace

    def import_data(self, df):
        self.data = df

    def optimise(self, parallelize=True):
        if self.method == "BruteForce":
            if isinstance(self.data, type(None)):
                params = self.parameter_searchspace
            else:
                params = [[self.data]] + self.parameter_searchspace

            if isinstance(self.metrics_searchspace, type(None)):
                pass
            else:
                params = params + self.metrics_searchspace

            return brute_force_optimizer(params, alpha=self.alpha_function, optim=self.optim_function, parallelize=parallelize)

        if self.method == "MCMC":
            mc = MCMC(alpha_fn=self.alpha_function, alpha_fn_params_0=self.guess, target=self.target,
                      num_iters=self.iters, prior=self.prior,
                      optimize_fn=self.optim_function, lower_limit=self.lower_limit, upper_limit=self.upper_limit)
            rs = mc.optimize()
            return mc, rs
