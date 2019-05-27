import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.metrics
import sklearn
import progressbar
import sklearn.model_selection
from plotnine import *
import pdb
import sys

sys.path.append("smooth_rf/")
import smooth_base
import smooth_level


# function

def average_depth(random_forest, data):
    """
    calculate the average depth of each point (average across trees)

    Arguments:
    ----------
    random_forest : sklearn random forest model (fit)
    data : array (n, p)
        data frame that can be predicted from random_forest

    Returns:
    --------
    average_depth : array (n,)
        vector of average depth in forest of each data point
    """

    # test:
    #rf_fit
    #smooth_rf_opt
    #d1 = average_depth(rf_fit, data)
    #d2 = average_depth(smooth_rf_opt, data)
    #np.all(d1 == d2)


    n_trees = len(random_forest.estimators_)
    n_obs = data.shape[0]
    depth = np.zeros(n_obs)
    for t in random_forest.estimators_:
        d_path = t.decision_path(data)
        depth = depth + np.array(d_path.sum(axis = 1)).ravel()

    return depth / n_trees



# start of analysis


data, y  = smooth_base.generate_data(large_n = 650)

data_vis = pd.DataFrame(data = {"x1":data[:,0],
                                "x2":data[:,1],
                                "y":y},
                        columns = ["x1","x2","y"])

ggout = ggplot(data_vis) +\
    geom_point(aes(x = "x1",y ="x2", color = "factor(y)")) +\
    theme_minimal() +\
    labs(x= "X1", y = "X2", color = "value (minus 100)")



rf = sklearn.ensemble.RandomForestRegressor(n_estimators = 300)
rf_fit = rf.fit(data,y)

smooth_rf_opt, smooth_rf_last ,_, _ = smooth_base.smooth(
                                rf_fit,
                                X_trained = data,
                                y_trained = y.ravel(),
                                X_tune = None,
                                y_tune = None,
                                resample_tune= False, # oob
                                no_constraint = False,
                                subgrad_max_num = 10000,
                                subgrad_t_fix = 1,
                                parents_all=True,
                                verbose = True,
                                all_trees = False,
                                initial_lamb_seed = None)

# test data
data_test, y_test  = smooth_base.generate_data(large_n = 10000)

reorder = np.random.choice(data_test.shape[0],
                           size = data_test.shape[0], replace= False)
data_test = data_test[reorder,:]
y_test = y_test[reorder]

yhat_base = rf_fit.predict(data_test)
yhat_smooth = smooth_rf_opt.predict(data_test)

base_mse = sklearn.metrics.mean_squared_error(y_true = y_test, y_pred = yhat_base)
smooth_mse = sklearn.metrics.mean_squared_error(y_true = y_test, y_pred = yhat_smooth)


error_base = np.abs(yhat_base - y_test)
error_smooth = np.abs(yhat_smooth - y_test)

extreme_binary = np.max([np.max(np.abs(error_base)),
                        np.max(np.abs(error_smooth))])

col_vis = error_base - error_smooth
extreme = np.max(np.abs(col_vis))

mean_depth_test = average_depth(rf_fit,data_test)

data_vis = pd.DataFrame(data = {"X1":data_test[:,0],
                                "X2":data_test[:,1],
                                "y": y_test.ravel(),
                                "error_base":error_base.copy(),
                                "error_smooth":error_smooth.copy(),
                                "error":col_vis.copy(),
                                "mean_depth":mean_depth_test.copy()},
                        columns = ["X1","X2","y","error",
                                   "error_base","error_smooth",
                                   "mean_depth"])


a = ggplot(data_vis) +\
  geom_point(aes(x = "X1", y="X2", color = "error"),
             size = .5) +\
  scale_color_continuous(name = "bwr",
                         limits= [-extreme, extreme]) +\
  theme_bw() +\
  labs(color = "Difference in Error",
       title = r'Difference in Error ($Error_{base} - Error_{smooth}$)')

b = ggplot(data_vis) +\
  geom_point(aes(x = "X1", y="X2", color = "error_base"),
             size = .5) +\
  scale_color_continuous(name = "binary",
                         limits= [0, extreme_binary]) +\
  theme_bw() +\
  labs(color = "Error",
       title = "Error from Base Random Forest")

c = ggplot(data_vis) +\
  geom_point(aes(x = "X1", y="X2", color = "error_smooth"),
             size = .5) +\
  scale_color_continuous(name = "binary",
                         limits= [0, extreme_binary]) +\
  theme_bw() +\
  labs(color = "Error",
       title = "Error from Smoothed Random Forest")

d = ggplot(data_vis) +\
  geom_point(aes(x = "X1", y="X2", color = "factor(y)"),
             size = .5) +\
  theme_bw() +\
  labs(color = "True Value (discrete)",
       title = "Test Set True Values")

e = ggplot(data_vis,aes(x = "mean_depth", y = "error")) +\
  geom_point(alpha = .1) +\
  theme_bw() +\
  labs(x = "Mean depth in Forest",
       y = "Difference in Error",
       title = "Lack of relationship between diff in errors and depth")

f = ggplot(data_vis, aes(x = "X1", y = "X2", color = "mean_depth")) +\
  geom_point() +\
  scale_color_continuous(name = "Blues") +\
  theme_bw() +\
  labs(color = "Mean depth in Forest",
       title = "Mean depth in Forest (Depth averaged across trees)")

g = ggplot(data_vis) +\
  geom_point(aes(x = "error_base", y = "error_smooth"),
             alpha = .05) +\
  geom_abline(intercept = 0, slope = 1) +\
  theme_bw() +\
  labs(x = "Error from Random Forest",
       y = "Error from Smooth Random Forest",
       title = "Comparing Errors Between Models",
       subtitle = r"(total error: rf: %f vs srf: %f)" %\
       (base_mse, smooth_mse))

save_as_pdf_pages([a + theme(figure_size = (8,6))],
                  filename = "images/diff_error"+"_understanding_smoothing.pdf")
save_as_pdf_pages([b + theme(figure_size = (8,6))],
                  filename = "images/error_base"+"_understanding_smoothing.pdf")
save_as_pdf_pages([c + theme(figure_size = (8,6))],
                  filename = "images/error_smooth"+"_understanding_smoothing.pdf")
save_as_pdf_pages([d + theme(figure_size = (8,6))],
                  filename = "images/truth"+"_understanding_smoothing.pdf")
save_as_pdf_pages([e + theme(figure_size = (8,6))],
                  filename = "images/mean_depth_diff_error"+"_understanding_smoothing.pdf")
save_as_pdf_pages([f + theme(figure_size = (8,6))],
                  filename = "images/mean_depth"+"_understanding_smoothing.pdf")
save_as_pdf_pages([g + theme(figure_size = (8,6))],
                  filename = "images/error_vs_error"+"_understanding_smoothing.pdf")


save_as_pdf_pages([a + theme(figure_size = (8,6)),
                  b + theme(figure_size = (8,6)),
                  c + theme(figure_size = (8,6)),
                  d + theme(figure_size = (8,6)),
                  e + theme(figure_size = (8,6)),
                  f + theme(figure_size = (8,6)),
                  g + theme(figure_size = (8,6))],
                  filename = "images/understanding_smoothing.pdf")


# some of these observations might be due to the decision on the values of the classes
# we'll see
