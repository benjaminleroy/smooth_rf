import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import sys, os

import smooth_rf

def _rosenbrock(vec, a=1, b=100):
    """
    Returns Rosenbrock function, a non-convex function, whose global minimum
    is at (a,a^2).

    f(x,y) = (a-x)^2 + b(y-x^2)^2

    Arguments:
    ----------
    vec : array (2,)
        2d array containing (x,y)
    a : scalar
    b : scalar

    Returns:
    --------
    rosenbrock : scalar
        rosenbrock value f(vec[0],vec[1] | a=a, b=b)

    Notes:
        https://en.wikipedia.org/wiki/Rosenbrock_function

    """
    return (a - vec[0])**2 + b * (vec[1] - vec[0]**2)**2
def _d_rosenbrock(vec, a=1, b=100):
    """
    Returns derivative of Rosenbrock function, a non-convex function, whose
    global minimum is at (a,a^2).

    f(x,y) = (a-x)^2 + b(y-x^2)^2
    grad f(x,y) = ( -2*(a-x) + 4bx(y-x^2), 2b(y-x^2) )

    Arguments:
    ----------
    vec : array
        2d array containing (x,y)
    a : scalar
    b : scalar

    Returns:
    --------
    grad : array (2,)
        gradient of rosenbrock grad f(vec[0],vec[1] | a=a, b=b)

    Notes:
        https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    grad = np.array([-2 * (a - vec[0]) - 4 * b * (vec[1]-vec[0]**2) * vec[0],
                    2 * b * (vec[1] - vec[0]**2)])
    return grad

def test_adam_step():
    """
    test for adam step

    similar to
    https://github.com/pytorch/pytorch/blob/master/test/optim/test.py
    """
    d_rosenbrock_lamb = lambda lamb: _d_rosenbrock(lamb,a = 1, b = 100)

    params = [{"alpha": 1e-4},
              {"alpha": 1e-4, "beta_1": 0.92},
              {"alpha": 1e-4, "beta_1": 0.92, "beta_2": 0.96},
              {"alpha": 1e-4, "beta_1": 0.92, "beta_2": 0.96, "eps": 1e-3}]


    for idx, param in enumerate(params):
        #print(param)
        #print("------")
        lamb = np.array([1.5,1.5])
        iv = None
        for step_idx in range(50000):
            lamb, iv = smooth_rf.adam_step(_d_rosenbrock, lamb,
                                 internal_values = iv, **param)
            if step_idx % 10000 == 0:
                pass
                #print('{:.8f}\t{:.8f}\t'.format(lamb[0], lamb[1]))

        if idx > 1:
            assert np.all(np.abs(lamb - np.array([1,1])) < 1e-3), \
                "adam step doesn't seem to coverge close enough to global " +\
                "optimal"
