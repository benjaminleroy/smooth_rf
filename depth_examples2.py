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

# reproducing example on page 114 (Microsoft)

np.random.seed(16)
select = 300

# check overfitting potential


def check_rf_grow(n_data, n_large, n_draws,
               reg_or_class="reg", depth_range=np.arange(1,50),
               verbose = True, ntree=1):
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

    score_mat = np.zeros((5, n_depth, n_draws))

    for i, max_depth in depth_iter:
        for j in np.arange(n_draws):
            # data generation
            ## data, y = smooth_base.generate_data(large_n=n_data)
            data, y, _ = smooth_base.generate_data_knn(n=n_data,
                                                       p=np.array([.3,.7]))
            y = y + 100
            # test
            ##data_test, y_test = smooth_base.generate_data(large_n=n_large)
            data_test, y_test, _ = smooth_base.generate_data_knn(n=n_large,
                                                        p=np.array([.3,.7]))

            y_test = y_test + 100

            model = model_type(max_depth=max_depth,n_estimators=ntree)
            model_fit = model.fit(data, y)

            # smooth equal weight
            smooth_ew_r_c, smooth_ew_r_c_last ,_, c1 = smooth_base.smooth(
                                model_fit,
                                X_trained = data,
                                y_trained = y.ravel(),
                                resample_tune=True,
                                no_constraint = False,
                                subgrad_max_num = 10000,
                                subgrad_t_fix = 1,
                                verbose = False,
                                all_trees = True)
            smooth_ew_r_nc, smooth_ew_r_nc_last, _, c2 = smooth_base.smooth(
                                model_fit,
                                X_trained = data,
                                y_trained = y.ravel(),
                                resample_tune=True,
                                no_constraint = True,
                                subgrad_max_num = 10000,
                                subgrad_t_fix = 1,
                                verbose = False,
                                all_trees = True)
            smooth_ew_ob_c, smooth_ew_ob_c_last, _, c3 = smooth_base.smooth(
                                model_fit,
                                X_trained = data,
                                y_trained = y.ravel(),
                                resample_tune=True,
                                no_constraint = False,
                                subgrad_max_num = 10000,
                                subgrad_t_fix = 1,
                                verbose = False,
                                all_trees = True)
            smooth_ew_ob_nc, smooth_ew_ob_nc_last, _, c4 = smooth_base.smooth(
                                model_fit,
                                X_trained = data,
                                y_trained = y.ravel(),
                                resample_tune=True,
                                no_constraint = False,
                                subgrad_max_num = 10000,
                                subgrad_t_fix = 1,
                                verbose = False,
                                all_trees = True)

            # # smooth level weight
            # smooth_lw_r_c = smooth_rf.smooth_all(
            #                     model_fit,
            #                     X_trained = data,
            #                     y_trained = y.ravel(),
            #                     resample_tune = True,
            #                     no_constraint = False,
            #                     verbose = False)

            # smooth_lw_r_nc = smooth_rf.smooth_all(
            #                     model_fit,
            #                     X_trained = data,
            #                     y_trained = y.ravel(),
            #                     resample_tune = True,
            #                     no_constraint = True,
            #                     verbose = False)

            # smooth_lw_ob_c = smooth_rf.smooth_all(
            #                     model_fit,
            #                     X_trained = data,
            #                     y_trained = y.ravel(),
            #                     resample_tune = False,
            #                     no_constraint = False,
            #                     verbose = False)

            # smooth_lw_ob_nc = smooth_rf.smooth_all(
            #                     model_fit,
            #                     X_trained = data,
            #                     y_trained = y.ravel(),
            #                     resample_tune = False,
            #                     no_constraint = True,
            #                     verbose = False)

            all_models = [model_fit,
                          smooth_ew_r_c, smooth_ew_r_nc,
                          smooth_ew_ob_c, smooth_ew_ob_nc,
                          smooth_ew_r_c_last, smooth_ew_r_nc_last,
                          smooth_ew_ob_c_last, smooth_ew_ob_nc_last,
                          # smooth_lw_r_c, smooth_lw_r_nc,
                          # smooth_lw_ob_c, smooth_lw_ob_nc
                          ]

            for m_idx, s_model in enumerate(all_models):
                yhat_test = s_model.predict(data_test)

                score_mat[m_idx, i,j] = scoring(y_test,yhat_test)

    return score_mat, [c1, c2, c3, c4]


def cv_vis(cv_mat, idx_range=None):
    """
    Arguments:
    ----------
    cv_mat : array (m, d, nfold)
        array of cross validation error values (assumed to
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
        labs(y = "Cross-Validation Error",
             x = "Index")

    return ggout, data_vis


create_figs = True
if create_figs:
    # single tree
    if select == 1:
        score_mat_reg1, c = check_rf_grow(650, 10000, n_draws = 20, ntree = 1,
                                       depth_range = np.arange(2,50,2))
        reg_vis1, data_vis = cv_vis(score_mat_reg1, np.arange(2,50,2))

        data_vis.to_csv("images/tree1_reg.csv")

        save_as_pdf_pages([reg_vis1 + labs(title = "1 tree, reg") +\
                           theme(figure_size = (8,6))],
                          filename = "images/tree1_reg_all.pdf")



    # 10 trees
    if select == 10:
        score_mat_reg, c = check_rf_grow(650, 10000, n_draws = 20, ntree = 10,
                                      depth_range = np.arange(2,50,2))
        reg_vis10, data_vis10 = cv_vis(score_mat_reg, np.arange(2,50,2))

        data_vis10.to_csv("images/tree10_reg.csv")

        save_as_pdf_pages([reg_vis10 + labs(title = "10 tree, reg") +\
                           theme(figure_size = (8,6))],
                         filename = "images/tree10_reg_all.pdf")

    # 300 trees
    if select == 300:
        score_mat_reg300, c = check_rf_grow(650, 10000, n_draws = 20, ntree = 300,
                                         depth_range = np.arange(2,50,2))
        reg_vis300, data_vis300 = cv_vis(score_mat_reg300, np.arange(2,50,2))

        data_vis300.to_csv("images/tree300_reg.csv")

        save_as_pdf_pages([reg_vis300 + labs(title = "300 tree, reg") +\
                           theme(figure_size = (8,6))],
                         filename = "images/tree300_reg_all.pdf")


