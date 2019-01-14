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

sys.path.append("smooth_rf/")
import smooth_base
import smooth_level as smooth_rf



# input
# 1: num_trees = [1, 10, 300] (integer)
# 2: tuning = ["resample", "oob", "oracle"]
# 3: constrained = ["c","nc"]
# 4: style = ["lb", "eb"]
# 5: distance = ["l","p"]
# next are only needed if s = "lb":
# 6: initial lambda = ["rf","r"]
# 7: batch = ["tree", "all"]
# 8: max_iter = 10000

n = sys.argv[1]
num_trees = np.int(n)

tuning = sys.argv[2]

c = sys.argv[3]
if c == "c":
    constrained = True
else if c == "nc":
    constrained = False
else:
    stop("c needs to be 1 of the 2 options")

s == sys.argv[4]
if s == "lb":
    style = "level-base"
else if s == "eb":
    style = "element-based"
else:
    stop("s needs to be 1 of the 2 options")


d = sys.argv[5]
if d == "l":
    parents_all = False
else if d == "p":
    parents_all = True
else:
    stop("d needs to be 1 of 2 options")

if s == "eb":
    i = sys.argv[6]
    if i == "rf":
        initial_lamb = "rf-init"
    else if i == "r":
        initial_lamb = "random-init"
    else:
        stop("i needs to be 1 of the 2 options")

    b = sys.argv[7]
    if b == "tree":
        batch = "single-tree"
    else if b == "all":
        batch = "all-trees"
    else:
        stop("b needs to be 1 of 2 options")

    m = sys.argv[8]
    max_iter = np.int(m)
else:
    initial_lamb = ""
    batch = ""
    max_iter = ""





# without multiple options currently
data_set = "microsoft"

# reproducing example on page 114 (Microsoft)

#np.random.seed(16)

# check overfitting potential


def get_random_seed():
    s = np.random.random(1) * 1000000
    s = np.int(s)
    return s

def check_rf_grow(n_data, n_large, n_draws,
               reg_or_class="reg", depth_range=np.arange(1,50),
               verbose = True, ntree=1,
               data_set = ["microsoft", "knn"],
               tuning = ["resample", "oob", "oracle"],
               constrained = True,
               style = ["level-base", "element-based"],
               parents_all = False,
               batch = ["single-tree", "all-trees"],
               initial_lamb = ["rf-init", "random-init"],
               max_iter = 10000):


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
        data_generator = smooth_base.generate_data
    else if data_set == "knn":
        data_generator = lambda large_n: smooth_base.generate_data_knn(
                                                     n=large_n,
                                                     p=np.array([.3,.7]))
    else:
        stop("data_set option needs to be 1 of the 2 options")

    if tuning == "resample":
        resample_input = True
    else:
        resample_input = False

    if initial_lamb == "rf-init":
        initial_lamb_seed_f = lambda : None
    else if initial_lamb == "random-init":
        initial_lamb_seed_f = get_random_seed

    if batch == "single-tree":
        all_trees = False
    else if batch == "all-trees":
        all_trees = True
    else:
        stop("batch option needs to be 1 of the 2 options")


    # storage devices
    if style == "level-base":
        score_mat = np.zeros((2, n_depth, n_draws))
        c_mat = None
    else if style == "element-based":
        score_mat = np.zeros((3, n_depth, n_draws))
        c_mat = np.zeros((depth_range.shape[0],n_draws,max_iter))
    else:
        stop("style needs to be 1 of the 2 options")



    for i, max_depth in depth_iter:
        for j in np.arange(n_draws):
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
            all_dat_test = data_generator(large_n=n_data)
            data_test, y_test = all_dat_test[0], all_dat_test[1]
            y_test = y_test + 100

            model = model_type(max_depth=max_depth,n_estimators=ntree)
            model_fit = model.fit(data, y)

            if style == "level-base":

                smooth_rf = smooth_rf.smooth_all(
                                    model_fit,
                                    X_trained = data,
                                    y_trained = y.ravel(),
                                    X_tune = data_tune,
                                    y_tune = y_tune,
                                    resample_tune = resample_input,
                                    no_constraint = not constrained,
                                    parents_all=parents_all,
                                    verbose = False)
                yhat_test_base = model_fit.predict(data_test)
                score_mat[0,i,j] = scoring(y_test,yhat_test_base)
                yhat_test = smooth_rf.predict(data_test)
                score_mat[1,i,j] = scoring(y_test,yhat_test)

            else if style ==  "element-based":

                smooth_rf_opt, smooth_rf_last ,_, c = smooth_base.smooth(
                                model_fit,
                                X_trained = data,
                                y_trained = y.ravel(),
                                X_tune = data_tune,
                                y_tune = y_tune,
                                resample_tune= resample_input,
                                no_constraint = not constrained,
                                subgrad_max_num = max_iter,
                                subgrad_t_fix = 10,
                                parents_all=parents_all,
                                verbose = False,
                                all_trees = all_trees,
                                initial_lamb_seed = initial_lamb_seed_f())
                c_mat[i, j, :] = c

                yhat_test_base = model_fit.predict(data_test)
                score_mat[0,i,j] = scoring(y_test,yhat_test_base)
                yhat_test_opt = smooth_rf_opt.predict(data_test)
                score_mat[1,i,j] = scoring(y_test,yhat_test_opt)
                yhat_test_last = smooth_rf_opt.predict(data_test)
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

    if idx_range is None:
        idx_range = np.arange(cv_mat.shape[1])

    data_vis = pd.DataFrame()
    for m_idx in np.arange(cv_mat.shape[0]):
        cv_mat_inner = cv_mat[m_idx,:,:]
        mu = cv_mat_inner.mean(axis = 1)
        sd = cv_mat_inner.std(axis = 1)

        data_vis_inner = pd.DataFrame(data = {"mu": mu,
                                        "lower": mu + sd,
                                        "upper": mu - sd,
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
                                                          dtype = np.int)
                                    "c": c_mat_inner},
                                    columns = ["depth", "sim","c"])

            data_vis = data_vis.append(data_vis_inner)

    ggout = ggplot(data_vis, aes(x = "step idx", y = "c",
                                 color = "sim")) +\
        geom_line() +\
        facet_wrap(facets="depth") +\
        labs(y = "Cost Function",
             x = "step of optimization")

    return ggout, data_vis



create_figs = True
if create_figs:
    if style == "level-base":
        score_mat, c = check_rf_grow(
               n_data=650,
               n_large=10000,
               n_draws=20,
               reg_or_class="reg",
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
               max_iter = max_iter)

        depth_vis, data_vis_depth = depth_error_vis(score_mat,
                                             np.arange(2,50,2))

        data_vis_depth.to_csv("images/data_vis_depth_"+\
                                 data_set + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c + "_" +\
                                 style + "_" +\
                                 initial_lamb + "_" +\
                                 batch + "_" +\
                                 str(max_iter) + ".csv")

        save_as_pdf_pages([depth_vis  +\
                           theme(figure_size = (8,6))],
                          filename = "images/depth_vis" +\
                                 data_set + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c + "_" +\
                                 style + "_" +\
                                 initial_lamb + "_" +\
                                 batch + "_" +\
                                 str(max_iter) + ".pdf")

        if style == "element-based":
            cost_vis, data_vis_cost = cost_vis(c, np.arange(2,50,2))

            data_vis_cost.to_csv("images/data_vis_cost_"+\
                                 data_set + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 initial_lamb + "_" +\
                                 batch + "_" +\
                                 str(max_iter) + ".csv")

            save_as_pdf_pages([cost_vis  +\
                           theme(figure_size = (8,6))],
                          filename = "images/cost_vis" +\
                                 data_set + "_" +\
                                 "trees" + str(num_trees) + "_" +\
                                 tuning + "_" +\
                                 c + "_" +\
                                 style + "_" +\
                                 "distance-"+ d +\
                                 initial_lamb + "_" +\
                                 batch + "_" +\
                                 str(max_iter) + ".pdf")




