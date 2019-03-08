import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import progressbar
import copy
import sklearn.ensemble
import sklearn
import pdb

def adam_step(grad_fun, lamb_init = None,
              alpha =.001,
              beta_1 = .9, beta_2 = .999,
              internal_values = None,
              eps = 1e-8):
    """
    Computes a single step of the Adam optimization

    Arguments:
    ----------
    grad_fun : function
        stocastic gradient function
    lamb_init : array
        initial values of lambda (if not None, overrides
        internal_values["lamb"])
    alpha, beta_1, beta_2 : scalars
        parameters relative to adam algorithm
    internal_values : dict
        dictionary of internal values that need to be saved (see returns)
    eps : scalar
        error scalar for division

    Returns:
    --------
    lamb : updated lamb value
    internal_values : dict
        updated internal values, updated t, lamb, and 1st and second moment of
        the gradient.

    Note:
    -----
    Adam paper:
        Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    """
    iv = internal_values

    if iv is None and lamb_init is None:
        NameError("lamb_init and internal_values can't both be None")

    if iv is None:
        iv = dict()
        iv["t"] = 0
        iv["lamb"] = lamb_init
        iv["1st moment"] = np.zeros(lamb_init.shape[0])
        iv["2nd moment"] = np.zeros(lamb_init.shape[0])

    if lamb_init is not None:
        iv["lamb"] = lamb_init

    iv["t"] += 1
    grad = grad_fun(iv["lamb"])
    iv["1st moment"] = iv["1st moment"] * beta_1 + (1 - beta_1) * grad
    iv["2nd moment"] = iv["2nd moment"] * beta_2 + (1 - beta_1) * (grad**2)
    hat_1st_moment = iv["1st moment"]/(1-(beta_1**iv["t"]))
    hat_2nd_moment = iv["2nd moment"]/(1-(beta_2**iv["t"]))

    iv["lamb"] = iv["lamb"] - alpha * hat_1st_moment/(hat_2nd_moment + eps)

    return iv["lamb"], iv





