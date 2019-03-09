import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn
import progressbar
import sklearn.model_selection
import sklearn.datasets
from plotnine import *
import pdb
import sys

#sys.path.append("../smooth_rf/")
#import smooth_base
#import smooth_level

import smooth_rf
path = "../"

# input
# 1: data_set = ["moon"]
# 2: num_trees = [1, 10, 300] (integer)
# 3: tuning = ["resample", "oob", "oracle"]
# 4: constrained = ["c","nc"]
# 5: style = ["lb", "eb"]
# 6: distance = ["l","p"]
# 7: inner_distance = ["standard", "max", "min"]
# 8: loss_function = ["ce","l2"]
# next are only needed if s = "lb":
# 9: initial lambda = ["rf","r"]
# 10: batch = ["tree", "all"]
# 11: max_iter = 10000
# 12: t = [.1,1,10,100] (scalar)

# ipython depth_run_class_adam.py moon 10 oob c eb p standard ce rf tree 10000 1
# ipython depth_run_class_adam.py microsoft 10 oob c eb p standard ce rf tree 10000 1


data_set = sys.argv[1]
if data_set != "moon" and data_set != "microsoft":
    NameError("dataset should be the moon or microsoft dataset")
else:
    reg_or_class = "class"
    y_all = None
    data_all = None

if data_set == "moon":
    n_data = 350
if data_set == "microsoft":
    n_data = 650

n = sys.argv[2]
num_trees = np.int(n)

tuning = sys.argv[3]

c_in = sys.argv[4]
if c_in == "c":
    constrained = True
elif c_in == "nc":
    constrained = False
else:
    NameError("c_in needs to be 1 of the 2 options")

s = sys.argv[5]
if s == "lb":
    style = "level-base"
elif s == "eb":
    style = "element-based"
else:
    NameError("s needs to be 1 of the 2 options")


d = sys.argv[6]
if d == "l":
    parents_all = False
elif d == "p":
    parents_all = True
else:
    NameError("d needs to be 1 of 2 options")

inner_dist = sys.argv[7]
if inner_dist not in ["standard", "max", "min"]:
    NameError("inner_dist, needs to be 1 of 3 options")

loss = sys.argv[8]
if loss not in ["ce","l2"]:
    NameError("loss need to be 1 of the 2 options")


if s == "eb":
    i = sys.argv[9]
    if i == "rf":
        initial_lamb = "rf-init"
    elif i == "r":
        initial_lamb = "random-init"
    else:
        NameError("i needs to be 1 of the 2 options")

    b = sys.argv[10]
    if b == "tree":
        batch = "single-tree"
    elif b == "all":
        batch = "all-trees"
    else:
        NameError("b needs to be 1 of 2 options")

    m = sys.argv[11]
    max_iter = np.int(m)

    subgrad_fix_t = np.float(sys.argv[12])

else:
    initial_lamb = ""
    batch = ""
    max_iter = ""
    subgrad_fix_t = ""


# check overfitting potential


def get_random_seed():
    s = np.random.random(1) * 1000000
    s = np.int(s)
    return s

def check_rf_grow(n_data, n_large, n_draws,
               reg_or_class="reg", depth_range=np.arange(1,50),
               verbose = True, ntree=1,
               data_set = ["microsoft", "knn", "online_news", "splice"],
               tuning = ["resample", "oob", "oracle"],
               constrained = True,
               style = ["level-base", "element-based"],
               parents_all = False,
               batch = ["single-tree", "all-trees"],
               initial_lamb = ["rf-init", "random-init"],
               loss_class = ["ce","l2"],
               max_iter = 10000, t = 1,
               data_all = None, y_all = None):

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

    # selection parameters
    if type(data_set) == list:
        data_set = data_set[0]
    if type(tuning) == list:
        tuning = tuning[0]
    if type(style) == list:
        style = style[0]
    if type(batch) == list:
        batch = batch[0]
    if type(initial_lamb) == list:
        initial_lamb = initial_lamb[0]

    if data_set == "microsoft":
        data_generator = smooth_rf.generate_data
    elif data_set == "knn":
        data_generator = lambda large_n: smooth_rf.generate_data_knn(
                                                     n=large_n,
                                                     p=np.array([.3,.7]))
    elif data_set == "moon":
        data_generator = lambda large_n: sklearn.datasets.make_moons(
                                                        n_samples=large_n,
                                                        noise=.3)
    elif data_set == "online_news" or data_set == "splice":
        if tuning == "oracle":
            NameError("tuning cannot be oracle for the online_news dataset")
        if data_all is None or y_all is None:
            NameError("data_all and y_all must be inserted when using the "+\
                      "online_news dataset")
    else:
        NameError("data_set option needs to be 1 of the 3 options")

    if tuning == "resample":
        resample_input = True
    else:
        resample_input = False

    if initial_lamb == "rf-init":
        initial_lamb_seed_f = lambda : None
    elif initial_lamb == "random-init":
        initial_lamb_seed_f = get_random_seed

    if batch == "single-tree":
        all_trees = False
    elif batch == "all-trees":
        all_trees = True
    else:
        NameError("batch option needs to be 1 of the 2 options")


    # storage devices
    if style == "level-base":
        score_mat = np.zeros((2, n_depth, n_draws))
        c_mat = None
    elif style == "element-based":
        score_mat = np.zeros((3, n_depth, n_draws))
        c_mat = np.zeros((depth_range.shape[0],n_draws,max_iter + 1))
    else:
        NameError("style needs to be 1 of the 2 options")

    if type(loss_class) is list:
        loss_class = loss_class[0]
    if loss_class not in ["ce","l2"]:
        NameError("style neds to be 1 of the 2 options")

    for i, max_depth in depth_iter:
        for j in np.arange(n_draws):

            if data_set == "online_news" or data_set == "splice":
                data, data_test, y, y_test = \
                    sklearn.model_selection.train_test_split(data_all,
                                                             y_all,
                                                             test_size = .5)
                data_tune = None
                y_tune = None
            else:
                # data generation
                all_dat = data_generator(large_n=n_data)
                data, y = all_dat[0], all_dat[1]
                y = y + 100

                # tune
                if tuning == "oracle":
                    all_dat_tune = data_generator(large_n=n_large)
                    data_tune, y_tune = all_dat_tune[0], all_dat_tune[1]
                    y_tune = y_tune + 100
                    y_tune = y_tune.ravel()
                else:
                    data_tune = None
                    y_tune = None

                # test
                all_dat_test = data_generator(large_n=n_large)
                data_test, y_test = all_dat_test[0], all_dat_test[1]
                y_test = y_test + 100

            model = model_type(max_depth=max_depth,n_estimators=ntree)
            model_fit = model.fit(data, y)

            if style == "level-base":

                smooth_rf_level = smooth_rf.smooth_all(
                                    model_fit,
                                    X_trained = data,
                                    y_trained = y.ravel(),
                                    X_tune = data_tune,
                                    y_tune = y_tune,
                                    resample_tune = resample_input,
                                    no_constraint = not constrained,
                                    parents_all=parents_all,
                                    verbose = False,
                                    )
                yhat_test_base = model_fit.predict(data_test)
                score_mat[0,i,j] = scoring(y_test,yhat_test_base)
                yhat_test = smooth_rf_level.predict(data_test)
                score_mat[1,i,j] = scoring(y_test,yhat_test)

            elif style ==  "element-based":
                smooth_rf_opt, smooth_rf_last ,_, c = smooth_rf.smooth(
                                model_fit,
                                X_trained = data,
                                y_trained = y.ravel(),
                                X_tune = data_tune,
                                y_tune = y_tune,
                                resample_tune= resample_input,
                                no_constraint = not constrained,
                                sgd_max_num = max_iter,
                                sgd_t_fix = t,
                                parents_all=parents_all,
                                verbose = False,
                                all_trees = all_trees,
                                initial_lamb_seed = initial_lamb_seed_f(),
                                adam = {"alpha": .001, "beta_1": .9,
                                        "beta_2": .999,"eps": 1e-8})
                c_mat[i, j, :] = c

                yhat_test_base = model_fit.predict(data_test)
                score_mat[0,i,j] = scoring(y_test,yhat_test_base)
                yhat_test_opt = smooth_rf_opt.predict(data_test)
                score_mat[1,i,j] = scoring(y_test,yhat_test_opt)
                yhat_test_last = smooth_rf_last.predict(data_test)
                score_mat[2,i,j] = scoring(y_test,yhat_test_last)

    return score_mat, c_mat


def depth_error_vis(cv_mat, idx_range=None):
    """
    Arguments:
    ----------
    cv_mat : array (m, d, nfold)
        array of cross validation error values (assumed to be
        averagable and that one can find it's variance)
    """


    n_sim = cv_mat.shape[1]

    if idx_range is None:
        idx_range = np.arange(cv_mat.shape[1])

    data_vis = pd.DataFrame()
    for m_idx in np.arange(cv_mat.shape[0]):
        cv_mat_inner = cv_mat[m_idx,:,:]
        mu = cv_mat_inner.mean(axis = 1)
        sd = cv_mat_inner.std(axis = 1)

        data_vis_inner = pd.DataFrame(data = {"mu": mu,
                                        "lower": mu + sd/np.sqrt(n_sim),
                                        "upper": mu - sd/np.sqrt(n_sim),
                                        "idx range": idx_range,
                                        "model name": str(m_idx)})

        data_vis = data_vis.append(data_vis_inner)

    ggout = ggplot(data_vis, aes(x = "idx range",
                                 color = "model name")) +\
        geom_line(aes(y = "mu")) +\
        geom_point(aes(y = "mu")) +\
        geom_ribbon(aes(ymin = "lower", ymax = "upper",
                        color= "model name"),alpha = .3) +\
        labs(y = "Test Error",
             x = "Index")

    return ggout, data_vis


def cost_vis(c_mat, depth_range=None):
    """
    Arguments:
    ----------
    c_mat : array (n_depth, n_sim, n_steps)
        array of cost values
    """
    if depth_range is None:
        depth_range = np.arange(c_mat.shape[1])

    data_vis = pd.DataFrame()
    for d_idx, depth_value in enumerate(depth_range):
        for s_idx in np.arange(c_mat.shape[1]):
            c_mat_inner = c_mat[d_idx,s_idx,:]

            data_vis_inner = pd.DataFrame(data =
                                    {"depth": depth_value,
                                    "sim": s_idx,
                                    "step idx": np.arange(c_mat_inner.shape[0],
                                                          dtype = np.int),
                                    "c": c_mat_inner},
                                    columns = ["depth", "sim","step idx","c"])

            data_vis = data_vis.append(data_vis_inner)

    ggout = ggplot(data_vis, aes(x = "step idx", y = "c",
                                 color = "factor(sim)")) +\
        geom_line(alpha = .1) +\
        facet_wrap(facets="depth") +\
        labs(y = "Cost Function",
             x = "step of optimization")

    return ggout, data_vis



create_figs = True
if create_figs:

    score_mat, c = check_rf_grow(
           n_data=n_data,
           n_large=5000,
           n_draws=20,
           reg_or_class=reg_or_class,
           depth_range=np.arange(2,50,2),
           verbose=True,
           ntree=num_trees,
           data_set = data_set,
           tuning = tuning,
           constrained = constrained,
           style = style,
           parents_all = parents_all,
           batch = batch,
           initial_lamb = initial_lamb,
           max_iter = max_iter, t = subgrad_fix_t,
           data_all = data_all, y_all = y_all,
           loss_class = loss
           )

    depth_vis, data_vis_depth = depth_error_vis(score_mat,
                                         np.arange(2,50,2))

    if style == "level-base":

        data_vis_depth.to_csv(path +\
                              "images/data_vis_depth_"+\
                                 data_set + "_" + reg_or_class + "_" +\
                                 "loss-" + loss + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c_in + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 "_tree_depth_dist-"+ inner_dist+\
                                 "_adam" +\
                                 ".csv")

        save_as_pdf_pages([depth_vis  +\
                           theme(figure_size = (8,6))],
                          filename = path +\
                                 "images/depth_vis_"  +\
                                 data_set + "_" + reg_or_class + "_" +\
                                 "loss-" + loss + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c_in + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 "_tree_depth_dist-"+ inner_dist+\
                                 "_adam" +\
                                  ".pdf")

    if style == "element-based":

        data_vis_depth.to_csv(path +\
                              "images/data_vis_depth_" +\
                                 data_set + "_" + reg_or_class + "_" +\
                                 "loss-" + loss + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c_in + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 "_tree_depth_dist-"+ inner_dist+ "_" +\
                                 "init_lamb-" + initial_lamb + "_" +\
                                 "fix_t-" + str(subgrad_fix_t) + "_" +\
                                 batch + "_" +\
                                 str(max_iter) +\
                                 "_adam" +\
                                 ".csv")

        save_as_pdf_pages([depth_vis  +\
                           theme(figure_size = (8,6))],
                          filename = path +\
                                 "images/depth_vis_" +\
                                 data_set + "_" + reg_or_class + "_" +\
                                 "loss-" + loss + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c_in + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 "_tree_depth_dist-"+ inner_dist+ "_" +\
                                 "init_lamb-" + initial_lamb + "_" +\
                                 "fix_t-" + str(subgrad_fix_t) + "_" +\
                                 batch + "_" +\
                                 str(max_iter) +\
                                 "_adam" +\
                                 ".pdf")



        cost_vis, data_vis_cost = cost_vis(c, np.arange(2,50,2))

        data_vis_cost.to_csv(path +\
                                 "images/data_vis_cost_" +\
                                 data_set + "_" + reg_or_class + "_" +\
                                 "loss-" + loss + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c_in + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 "_tree_depth_dist-"+ inner_dist+ "_" +\
                                 "init_lamb-" + initial_lamb + "_" +\
                                 "fix_t-" + str(subgrad_fix_t) + "_" +\
                                 batch + "_" +\
                                 str(max_iter) +\
                                 "_adam" +\
                                 ".csv")

        save_as_pdf_pages([cost_vis +\
                           theme(figure_size = (8,6))],
                          filename = path +\
                                 "images/cost_vis_" +\
                                 data_set + "_" + reg_or_class + "_" +\
                                 "loss-" + loss + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c_in + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 "_tree_depth_dist-"+ inner_dist+ "_" +\
                                 "init_lamb-" + initial_lamb + "_" +\
                                 "fix_t-" + str(subgrad_fix_t) + "_" +\
                                 batch + "_" +\
                                 str(max_iter) +\
                                 "_adam" +\
                                 ".pdf")





