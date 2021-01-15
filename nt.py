import jax.numpy as np

from jax import random

import neural_tangents as nt
from neural_tangents import stax



class NNGPRegression(object):
    def __init__(self, input_x, output_y, noise_var=1e-4, nonlinearity='relu', n_depth =2):
        self.input_x = input_x
        self.output_y = output_y
        self.noise_var = noise_var

        if nonlinearity == 'relu':
            act_func = stax.Relu
        elif nonlinearity == 'tanh':
            act_func = stax.Erf

        self.init_fn, self.apply_fn, self.kernel_fn = stax.serial(
            stax.Dense(512), act_func(),
            stax.Dense(512), act_func(),
            stax.Dense(1)
        )
        self.predict_fn = nt.predict.gradient_descent_mse_ensemble(self.kernel_fn, self.input_x,
                                                      self.output_y, diag_reg=self.noise_var)

    
    def predict(self, test_x, get_var=False):
        if get_var:
            nngp_mean, nngp_covariance = self.predict_fn(x_test=test_x, get='nngp', compute_cov=True)
            return nngp_mean, nngp_covariance
        else:
            nngp_mean = self.predict_fn(x_test=test_x, get='nngp', compute_cov=False)
            return nngp_mean
