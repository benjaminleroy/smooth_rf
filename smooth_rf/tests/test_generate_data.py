import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import progressbar
import copy
import sklearn.ensemble
import sklearn
import pdb

import smooth_rf


def test_generate_data():
    """
    test generate_data (basic structure)
    """

    data, y = smooth_rf.generate_data(large_n = 100)

    assert y.shape[0] < 100 * 1.1 and y.shape[0] > 100 * .9, \
        "number of observations is not close to that requested"

    assert data.shape[0] == y.shape[0], \
        "output of generate_data function should have same number of rows"

    assert data.shape[1] == 2, \
        "the first output of generate_data shoudl be a 2d array"

def test_generate_data_knn():
    """
    test generate_data_knn (basic structure)
    """

    data, y, values = smooth_rf.generate_data_knn(n = 100,
                                                  p = np.array([.5,.5]))

    assert y.shape[0] == 100, \
        "number of observations not equal to that requested"

    assert data.shape[0] == y.shape[0], \
        "output of generate_data function should have same number of rows"

    assert values.shape[0] == y.shape[0], \
        "output of generate_data function should have same number of rows"

    assert np.max(values) == 3 and np.min(values) == 0 \
        and np.all(np.ceil(values) == values), \
        "values is not constrainted within the correct number of integers"

    assert data.shape[1] == 2, \
        "the first output of generate_data shoudl be a 2d array"


def test_spirals():
    """
    test spirals
    """
    data_all = smooth_rf.spirals(n_total = 90, n_classes = 3,
                                 noise_sd = .1, t_shift = 0)

    assert data_all.shape[0] == 90, \
        "number of observations not equal to that requested"

    assert np.max(data_all["class"]) == 2 and np.min(data_all["class"]) == 0 \
        and np.all(np.ceil(data_all["class"]) == data_all["class"]), \
        "values is not constrainted within the correct number of integers"

    assert data_all.shape[1] == 4, \
        "the first output of generate_data should have 4 columns"
