import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn
import progressbar
import sklearn.model_selection
from plotnine import *
import pdb
import sys

import smooth_rf


path = "../"


vis = False

if vis:
    # all histograms

    plt.figure()
    data_all.hist()

    plt.figure()
    y_all.hist()

    # correlation heatmap
    plt.figure()
    correlations = data.corr()
    plt.matshow(correlations,vmin = -1, vmax = 1)


# random forest...
def regression_prep(data_train, data_test, y_train, y_test,
                    depth_range = np.arange(2,50,2),
                    reg_or_class="reg",
                    verbose = True, ntree=1#,
                    # tuning = ["resample", "oob", "oracle"],
                    # constrained = True,
                    # style = ["level-base", "element-based"],
                    # parents_all = False,
                    # batch = ["single-tree", "all-trees"],
                    # initial_lamb = ["rf-init", "random-init"],
                    # max_iter = 10000
                    ):
    model_type = None
    if reg_or_class == "reg":
        model_type = sklearn.ensemble.RandomForestRegressor
        scoring = sklearn.metrics.mean_squared_error

    if reg_or_class == "class":
        model_type = sklearn.ensemble.RandomForestClassifier
        scoring = sklearn.metrics.accuracy_score

    if model_type is None:
        raise ValueError("reg_or_class input string is not 'reg' "+\
                         "nor 'class'.")

    depth_iter = list(enumerate(depth_range))
    n_depth = depth_range.shape[0]

    if verbose:
        bar = progressbar.ProgressBar()
        depth_iter = bar(depth_iter)

    score_vec = np.zeros(depth_range.shape[0])

    for i, max_depth in depth_iter:
        model = model_type(max_depth=max_depth,n_estimators=ntree)
        model_fit = model.fit(data_train, y_train)

        yhat_test_base = model_fit.predict(data_test)
        score_vec[i] = scoring(y_test,yhat_test_base)

    return score_vec

def depth_error_vis(cv_mat, idx_range=None):
    """
    Arguments:
    ----------
    cv_mat : array (m, d, nfold)
        array of cross validation error values (assumed to be
        averagable and that one can find it's variance)
    """

    n_sim = cv_mat.shape[0]

    if idx_range is None:
        idx_range = np.arange(cv_mat.shape[1])

    mu = cv_mat.mean(axis = 0)
    sd = cv_mat.std(axis = 0)

    data_vis = pd.DataFrame(data = {"mu": mu,
                                    "lower": mu + sd/np.sqrt(n_sim),
                                    "upper": mu - sd/np.sqrt(n_sim),
                                    "idx range": idx_range})


    ggout = ggplot(data_vis, aes(x = "idx range")) +\
        geom_line(aes(y = "mu")) +\
        geom_point(aes(y = "mu")) +\
        geom_ribbon(aes(ymin = "lower", ymax = "upper"),
                    alpha = .3) +\
        labs(y = "Test Error",
             x = "Index")

    return ggout, data_vis



reg_or_class = "class"
data_name = "spiral3_" + reg_or_class
n_sim = 100
depth_range = np.arange(2,50,2)
np.random.seed(100)
verbose = True

# n_tree = 10

n_tree = 10
score_mat = np.zeros((n_sim, depth_range.shape[0]))

if verbose:
    bar = progressbar.ProgressBar()
    sim_iter = bar(np.arange(n_sim))

for s_idx in sim_iter:
    data_all = smooth_rf.spirals(n_total = 3000, noise_sd = .1)
    y_all = data_all["class"]
    data_all.drop(columns = ["class", "t"])
    #data_all = data_all + 100
    y_all = y_all.ravel()


    data_train, data_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(data_all,
                                                 y_all,
                                                 test_size = .8)

    score_vec = regression_prep(data_train, data_test, y_train, y_test,
                      depth_range = np.arange(2,50,2),
                      reg_or_class=reg_or_class,
                      verbose = False, ntree=n_tree)
    score_mat[s_idx,:] = score_vec

score_mat10 = score_mat


ggout, data_vis = depth_error_vis(score_mat10, depth_range)

save_as_pdf_pages([ggout +\
                   theme(figure_size = (8,6))],
                  filename = path +  "images/depth_vis_" +\
                         data_name+"_tree" +\
                         str(n_tree)+\
                         ".pdf")

data_vis.to_csv(path + "images/data_vis_depth_" +\
                data_name+"_tree" +\
                str(n_tree) +\
                ".csv")

# n_tree = 300

n_tree = 300
score_mat = np.zeros((n_sim, depth_range.shape[0]))


if verbose:
    bar = progressbar.ProgressBar()
    sim_iter = bar(np.arange(n_sim))


for s_idx in sim_iter:
    data_all = smooth_rf.spirals(n_total = 3000, noise_sd = .1)
    y_all = data_all["class"]
    data_all.drop(columns = ["class", "t"])
    #data_all = data_all + 100
    y_all = y_all.ravel()
    data_train, data_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(data_all,
                                                 y_all,
                                                 test_size = .8)
    score_vec = regression_prep(data_train, data_test, y_train, y_test,
                      depth_range = np.arange(2,50,2),
                      reg_or_class=reg_or_class,
                      verbose = False, ntree=n_tree)
    score_mat[s_idx,:] = score_vec


score_mat300 = score_mat

ggout300, data_vis300 = depth_error_vis(score_mat10,depth_range )


save_as_pdf_pages([ggout300 +\
                   theme(figure_size = (8,6))],
                  filename = path + "images/depth_vis_" +\
                         data_name+"_tree" +\
                         str(n_tree)+\
                         ".pdf")

data_vis300.to_csv(path + "images/data_vis_depth_" +\
                data_name+"_tree" +\
                str(n_tree) +\
                ".csv")
