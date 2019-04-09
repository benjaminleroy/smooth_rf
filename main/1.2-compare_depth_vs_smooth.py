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

def depth_search(X_train, y_train, depth_range = np.arange(2,50,2),
                 n_estimators = 100,
                 type = "reg", verbose = True):
    """
    recommend larger number of trees for oob error to stabilize
    """
    if type=="reg":
        model = sklearn.ensemble.RandomForestRegressor
    elif type == "class":
        model = sklearn.ensemble.RandomForestClassifier
    else:
        NameError("type must be 1 of the 2 options")


    first_iter = depth_range
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)

    best_depth = np.inf
    if type == "reg":
        best_oob_score = np.inf
    else:
        best_oob_score = -np.inf

    oob_cost = list()

    for depth in first_iter:
        rf = model(n_estimators = n_estimators,
                   max_depth = depth,
                   oob_score = True)

        rf = rf.fit(X = np.array(X_train),
                    y = y_train)

        oob_score = 1 - rf.oob_score_ # why would you do this to me...

        oob_cost.append(oob_score)

        if True: #type == "reg":
            if oob_score < best_oob_score:
                best_oob_score = oob_score
                best_depth = depth
        else:
            if oob_score > best_oob_score:
                best_oob_score = oob_score
                best_depth = depth

    return best_depth, best_oob_score, oob_cost, depth_range

def test_depth_search():
    # data creation
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))

    b_depth = depth_search(X_train, y_train)


def node_size_search(X_train, y_train, node_size_range = np.arange(2,50,2),
                     n_estimators = 100,
                     type = "reg", verbose = True):
    """
    recommend larger number of trees for oob error to stabilize
    """
    if type=="reg":
        model = sklearn.ensemble.RandomForestRegressor
    elif type == "class":
        model = sklearn.ensemble.RandomForestClassifier
    else:
        NameError("type must be 1 of the 2 options")


    first_iter = node_size_range
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)

    best_node_size = np.inf
    if type == "reg":
        best_oob_score = np.inf
    else:
        best_oob_score = -np.inf

    oob_cost = list()

    for node_size in first_iter:
        rf = model(n_estimators = n_estimators,
                   min_samples_split = np.int(node_size),
                   oob_score = True)

        rf = rf.fit(X = np.array(X_train),
                    y = y_train)

        oob_score = 1 - rf.oob_score_ # why would you do this to me...

        oob_cost.append(oob_score)

        if True: #type == "reg":
            if oob_score < best_oob_score:
                best_oob_score = oob_score
                best_node_size = node_size
        else:
            if oob_score > best_oob_score:
                best_oob_score = oob_score
                best_node_size = node_size

    return best_node_size, best_oob_score, oob_cost, node_size_range


def test_node_size_search():
    # data creation
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))

    b_node_size = node_size_search(X_train, y_train)




############
# Analysis #
############

# sys.argv = ["hi", "microsoft"]

# input
# 1: data_set = ["microsoft", "online_news"]

data_set = sys.argv[1]
if data_set == "online_news":
    data = pd.read_csv(path +\
                       "data/OnlineNewsPopularity/OnlineNewsPopularity.csv")

    y_all = np.log10(data[" shares"])
    data_all = data.drop(columns = ["url"," timedelta"," shares"])
    data_all = np.array(data_all)
elif data_set == "microsoft":
    data_generator = smooth_rf.generate_data
else:
    NameError("data_set needs to be 1 of the 2 options")



sim_amount = 20
n_data = 650
n_large = 10000
n_estimators = 100
n_steps = 10000
scoring = sklearn.metrics.mean_squared_error
depth_range = np.arange(2,50,2)
node_size_range = np.arange(2,50,2)
first_iter = np.arange(sim_amount)

bar = progressbar.ProgressBar()
first_iter = bar(first_iter)


sim_info = np.zeros((sim_amount, 6))

oob_mat = np.zeros((sim_amount, depth_range.shape[0]))
n_oob_mat = np.zeros((sim_amount, node_size_range.shape[0]))

for sim_idx in first_iter:

    # data simulation
    if data_set == "online_news":
        data, data_test, y, y_test = \
                    sklearn.model_selection.train_test_split(data_all,
                                                             y_all,
                                                             test_size = .5)
    else:
        all_dat = data_generator(large_n=n_data)
        data, y = all_dat[0], all_dat[1]
        y = y + 100

        all_dat_test = data_generator(large_n=n_large)
        data_test, y_test = all_dat_test[0], all_dat_test[1]
        y_test = y_test + 100


    # depth_selection:
    d_best, d_value, oob_vec, depth_vec = depth_search(data, y,
                                   depth_range = depth_range,
                                   n_estimators = n_estimators,
                                   type = "reg", verbose = False)

    oob_mat[sim_idx,:] = oob_vec

    d_rf = sklearn.ensemble.RandomForestRegressor(
                        n_estimators = n_estimators,
                        max_depth = d_best)
    d_rf_fit = d_rf.fit(data,y)
    yhat_test_depth = d_rf_fit.predict(data_test)


    # node_size_selection:
    n_best, n_value, n_oob_vec, node_vec = node_size_search(data, y,
                                   node_size_range = node_size_range,
                                   n_estimators = n_estimators,
                                   type = "reg", verbose = False)

    n_oob_mat[sim_idx,:] = n_oob_vec

    n_rf = sklearn.ensemble.RandomForestRegressor(
                        n_estimators = n_estimators,
                        min_samples_split = np.int(n_best))
    n_rf_fit = n_rf.fit(data,y)
    yhat_test_node = n_rf_fit.predict(data_test)


    # smoothed
    rf_base = sklearn.ensemble.RandomForestRegressor(
                        n_estimators = n_estimators)
    rf_fit = rf_base.fit(data,y)
    yhat_test_base = rf_fit.predict(data_test)


    rf_smooth, rf_smooth_b, _, _ = smooth_rf.smooth(
                                          random_forest = rf_fit,
                                          X_trained = data,
                                          y_trained = y,
                                          subgrad_max_num = n_steps,
                                          verbose = False,
                                          parents_all = True,
                                          distance_style = "standard",
                                          no_constraint = False
                                          )
    yhat_test_smooth = rf_smooth.predict(data_test)
    yhat_test_smooth_b = rf_smooth_b.predict(data_test)

    # depth_smoothed
    d_rf_smooth, d_rf_smooth_b, _, _ = smooth_rf.smooth(d_rf_fit,
                                          X_trained = data,
                                          y_trained = y,
                                          subgrad_max_num = n_steps,
                                          verbose = False,
                                          parents_all = True,
                                          distance_style = "standard",
                                          no_constraint = False)
    yhat_test_smooth_depth = d_rf_smooth.predict(data_test)
    yhat_test_smooth_depth_b = d_rf_smooth_b.predict(data_test)


    sim_info[sim_idx,:] = [scoring(y_test,yhat_test_depth),
                           scoring(y_test,yhat_test_base),
                           scoring(y_test,yhat_test_smooth),
                           scoring(y_test,yhat_test_smooth_depth),
                           scoring(y_test,yhat_test_smooth_b),
                           scoring(y_test,yhat_test_smooth_depth_b),
                           scoring(y_test, yhat_test_node)]



# End processing
np.savetxt(fname = path + "images/" + data_set + ".csv", X = sim_info)
np.savetxt(fname = path + "images/" + data_set + "_oob" + ".csv",
           X = oob_mat)

# Visualizing with ggplot


# accuracy of models
# ------------------
sim_info_pd = pd.DataFrame(sim_info, columns = ["depth selection", "base rf",
                                                "smooth opt lamb",
                                                "smooth opt lamb depth grown",
                                                "smooth final lamb",
                                                "smooth final lamb depth grown"
                                                ])

sim_vis_pd = sim_info_pd[["depth selection", "base rf",
             "smooth opt lamb"]]

sim_vis_pd["idx"] = pd.Series(np.arange(sim_vis_pd.shape[0],dtype = np.int))

sim_vis_pd_melt = sim_vis_pd.melt(id_vars = "idx")
sim_vis_pd_melt["variable"] = pd.Categorical(sim_vis_pd_melt["variable"],
                                             ordered=True,
                                             categories = ["base rf",
                                                           "depth selection",
                                                           "smooth opt lamb"])

ggvis = ggplot(sim_vis_pd_melt,
       aes(x = "variable", y = "value")) +\
  geom_line(aes(group = "idx"), alpha = .1) +\
  geom_boxplot(width = .5) +\
  labs(x = "Model",
       y = "Test Error",
       title = "Comparing Depth Selection vs Smoothing "+\
               "on 20 Samples of 'Microsoft Data'") +\
  theme_minimal()

save_as_pdf_pages([ggvis  +\
                   theme(figure_size = (8,6))],
                  filename = path +\
                         "images/depth_vs_smooth_microsoft" +\
                          ".pdf")


# depth model oob error
# ---------------------
depth_info_pd = pd.DataFrame(oob_mat,
                             columns = np.arange(2,50,2, dtype = np.int))
depth_info_pd["idx"] = pd.Series(np.arange(depth_info_pd.shape[0],
                                           dtype = np.int))
depth_info_pd = depth_info_pd.melt(id_vars = "idx")


ggvis_depth = ggplot(depth_info_pd) +\
  geom_line(aes(x = "variable", y = "value", group = "idx")) +\
  labs(x = "Maximum Depth",
       y = "OOB Error",
       title = "Examining OOB Error for Depth-Constrained RF (100 trees)")+\
  theme_minimal()

save_as_pdf_pages([ggvis_depth  +\
                   theme(figure_size = (8,6))],
                  filename = path +\
                         "images/depth_oob_microsoft" +\
                          ".pdf")



