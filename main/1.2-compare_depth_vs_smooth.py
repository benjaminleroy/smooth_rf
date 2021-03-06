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
    # if type == "reg":
    best_oob_score = np.inf
    # else:
    #     best_oob_score = -np.inf

    oob_cost = list()

    for depth in first_iter:
        rf = model(n_estimators = n_estimators,
                   max_depth = depth,
                   oob_score = True)

        rf = rf.fit(X = np.array(X_train),
                    y = y_train)

        oob_score = 1 - rf.oob_score_ # why would you do this to me...

        oob_cost.append(oob_score)

        if True:#type == "reg":
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


    b_depth = depth_search(X_train, y_train, type = "class")




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
    # if type == "reg":
    best_oob_score = np.inf
    # else:
    #     best_oob_score = -np.inf

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


# input
# 1: data_set = ["microsoft", "online_news", "titanic", "prgeng"]
# 2: oob_step =  ["false", "true"]
data_set = sys.argv[1]
if data_set == "online_news":
    data = pd.read_csv(path +\
                       "data/OnlineNewsPopularity/OnlineNewsPopularity.csv")

    y_all = np.log10(data[" shares"])
    data_all = data.drop(columns = ["url"," timedelta"," shares"])
    data_all = np.array(data_all)

    reg_or_class = "reg"
    data_set_name = "Online News"
elif data_set == "microsoft":
    reg_or_class = "reg"
    data_generator = smooth_rf.generate_data
    n_data = 650
    n_large = 10000
    data_set_name = "Microsoft"
elif data_set == "titanic":
    reg_or_class = "class"
    data_set_name = "Titanic"

    data_train = pd.read_csv(path + "data/titanic/titanic3.csv")

    data_train.pop("cabin")
    data_train.pop("name")
    data_train.pop("ticket")
    data_train.pop("body")
    data_train.pop("boat")
    data_train.pop("home.dest")
    data_train["pclass"] = data_train["pclass"].apply(str)

    NAs = pd.concat([data_train.isnull().sum()], axis=1)
    # NAs[NAs.sum(axis=1) > 0]




    # Filling missing Age values with mean
    data_train["age"] = data_train["age"].fillna(data_train["age"].mean())
    # Filling missing Embarked values with most common value
    data_train["embarked"] = data_train["embarked"].fillna(data_train["embarked"].mode()[0])


    for col in data_train.dtypes[data_train.dtypes == "object"].index:
        for_dummy = data_train.pop(col)
        data_train = pd.concat([data_train,
                                pd.get_dummies(for_dummy, prefix=col)],
                               axis=1)

    data_train = data_train.dropna()

    y_all = data_train.survived
    data_train.pop("survived")

    data_all = data_train

    data_all = np.array(data_all)
    y_all = y_all.ravel()
elif data_set == "prgeng":
    data_all = pd.read_csv(path + "data/prgeng/prgeng.txt", sep= " ")
    y_all = data_all["wageinc"]
    data_all.pop("wageinc")

    data_all = np.array(data_all)
    y_all = y_all.ravel()

    reg_or_class = "reg"
    data_set_name = "Engineer Salaries"
else:
    NameError("data_set needs one of the specified options")


oob_step_input = sys.argv[2]
if oob_step_input == "false":
    oob_step = False
else:
    oob_step = True



if reg_or_class == "class":
    model = sklearn.ensemble.RandomForestRegressor
    scoring = sklearn.metrics.accuracy_score

else:
    model = sklearn.ensemble.RandomForestClassifier
    scoring = sklearn.metrics.mean_squared_error


sim_amount = 50
n_estimators = 100
n_steps = 10000
depth_range = np.arange(2,50,2)
node_size_range = np.arange(2,50,2)
first_iter = np.arange(sim_amount)



bar = progressbar.ProgressBar()
first_iter = bar(first_iter)

sim_info = np.zeros((sim_amount, 7))

if oob_step:
    oob_mat = np.zeros((sim_amount, depth_range.shape[0]))
    n_oob_mat = np.zeros((sim_amount, node_size_range.shape[0]))

for sim_idx in first_iter:

    # data simulation
    if data_set == "online_news" or \
        data_set == "titanic" or \
        data_set == "prgeng":
        data, data_test, y, y_test = \
                    sklearn.model_selection.train_test_split(data_all,
                                                             y_all,
                                                             test_size = .5)
    else: # microsoft
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
                                   type = reg_or_class, verbose = False)

    if oob_step:
        oob_mat[sim_idx,:] = oob_vec

    d_rf = model(n_estimators = n_estimators,
                 max_depth = d_best)
    d_rf_fit = d_rf.fit(data,y)
    yhat_test_depth = d_rf_fit.predict(data_test)


    # node_size_selection:
    n_best, n_value, n_oob_vec, node_vec = node_size_search(data, y,
                                   node_size_range = node_size_range,
                                   n_estimators = n_estimators,
                                   type = "reg", verbose = False)

    if oob_step:
        n_oob_mat[sim_idx,:] = n_oob_vec

    n_rf = model(n_estimators = n_estimators,
                 min_samples_split = np.int(n_best))
    n_rf_fit = n_rf.fit(data,y)
    yhat_test_node = n_rf_fit.predict(data_test)


    # smoothed
    rf_base = model(n_estimators = n_estimators)
    rf_fit = rf_base.fit(data,y)
    yhat_test_base = rf_fit.predict(data_test)


    rf_smooth, rf_smooth_b, _, _ = smooth_rf.smooth(
                                          random_forest = rf_fit,
                                          X_trained = data,
                                          y_trained = y,
                                          sgd_max_num = n_steps,
                                          verbose = False,
                                          parents_all = True,
                                          distance_style = "standard",
                                          no_constraint = False,
                                          adam = {"alpha": .001, "beta_1": .9,
                                                  "beta_2": .999,"eps": 1e-8})
    yhat_test_smooth = rf_smooth.predict(data_test)
    yhat_test_smooth_b = rf_smooth_b.predict(data_test)

    # depth_smoothed
    d_rf_smooth, d_rf_smooth_b, _, _ = smooth_rf.smooth(d_rf_fit,
                                          X_trained = data,
                                          y_trained = y,
                                          sgd_max_num = n_steps,
                                          verbose = False,
                                          parents_all = True,
                                          distance_style = "standard",
                                          no_constraint = False,
                                          adam = {"alpha": .001, "beta_1": .9,
                                                  "beta_2": .999,"eps": 1e-8})
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
np.savetxt(fname = path + "images/" + data_set + "_adam.csv", X = sim_info)
if oob_step:
    np.savetxt(fname = path + "images/" + data_set + "_oob_adam" + ".csv",
               X = oob_mat)

# Visualizing with ggplot


# accuracy of models
# ------------------
sim_info_pd = pd.DataFrame(sim_info, columns = ["depth selection",
                                                "base rf",
                                                "smooth opt lamb",
                                                "smooth opt lamb depth grown",
                                                "smooth final lamb",
                                                "smooth final lamb depth grown",
                                                "node size selection"
                                                ])

sim_vis_pd = sim_info_pd[["depth selection", "base rf",
             "smooth opt lamb", "node size selection"]]

sim_vis_pd["idx"] = pd.Series(np.arange(sim_vis_pd.shape[0],dtype = np.int))

sim_vis_pd_melt = sim_vis_pd.melt(id_vars = "idx")
sim_vis_pd_melt["variable"] = pd.Categorical(sim_vis_pd_melt["variable"],
                                             ordered=True,
                                             categories = ["base rf",
                                                           "depth selection",
                                                           "smooth opt lamb",
                                                           "node size selection"])

ggvis = ggplot(sim_vis_pd_melt,
       aes(x = "variable", y = "value")) +\
  geom_line(aes(group = "idx"), alpha = .1) +\
  geom_boxplot(width = .5) +\
  labs(x = "Model",
       y = "Test Error",
       title = "Comparing Depth Selection vs Smoothing "+\
               "on 20 Samples of " + data_set_name +" Data'") +\
  theme_minimal()

save_as_pdf_pages([ggvis  +\
                   theme(figure_size = (8,6))],
                  filename = path +\
                         "images/depth_vs_smooth_" + data_set +\
                          "_adam.pdf")


# depth model oob error
# ---------------------
if oob_step:
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
                             "images/depth_oob_" + data_set +\
                              "adam_.pdf")



