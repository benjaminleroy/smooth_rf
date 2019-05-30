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
import copy
import pickle
import itertools

import smooth_rf

path = "../"

data_set = sys.argv[1]
reg_or_class = sys.argv[2]
n_trees = sys.argv[3]

def pull_data(data_set, path, reg_or_class="reg"):
    if data_set == "microsoft":
        n_data = 650*2
        X, y = smooth_rf.generate_data(large_n = n_data)

        if reg_or_class == "reg":
            y = y + 100

        return X, y
    elif data_set == "moon":
        n_data = 350*2
        X, y = sklearn.datasets.make_moons(n_samples=n_data,
                                           noise=.3)
        if reg_or_class == "reg":
            y = y + 100

        return X, y

    elif data_set == "prgeng":
        data_all = pd.read_csv(path + "data/prgeng/prgeng.txt", sep= " ")
        y_all = data_all["wageinc"]
        data_all.pop("wageinc")

        X = np.array(data_all)
        y = y_all.ravel()

        if reg_or_class != "reg":
            ValueError("must use 'reg' with 'prgeng' dataset")

        return X, y

    elif data_set == "titantic":
        data_train = pd.read_csv(path + "data/titanic/titanic3.csv")

        data_train.pop("cabin")
        data_train.pop("name")
        data_train.pop("ticket")
        data_train.pop("body")
        data_train.pop("boat")
        data_train.pop("home.dest")
        data_train["pclass"] = data_train["pclass"].apply(str)

        NAs = pd.concat([data_train.isnull().sum()], axis=1)

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

        X = np.array(data_all)
        y = y_all.ravel()

        if reg_or_class != "class":
            ValueError("must use 'class' with 'titanic' dataset")

        return X, y

def get_random_seed():
    s = np.random.random(1) * 1000000
    s = np.int(s)
    return s

def split_data(X,y, test_size=.5):
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y,
                                                 test_size=test_size)

    return X_train, y_train, X_test, y_test

def generate_rf(X_train, y_train, n_trees, reg_or_class="reg"):

    if reg_or_class == "reg":
        model_type = sklearn.ensemble.RandomForestRegressor
    elif reg_or_class == "class":
        model_type = sklearn.ensemble.RandomForestClassifier
    else:
        ValueError("reg_or_class must either be 'reg' or 'class'")

    model = model_type(n_estimators=n_trees)

    model_fit = model.fit(X_train, y_train)

    return model_fit


def assess_rf(random_forest, X_test, y_test):
    if type(random_forest) == sklearn.ensemble.RandomForestRegressor:
        scoring = sklearn.metrics.mean_squared_error
    elif type(random_forest) == sklearn.ensemble.RandomForestClassifier
        scoring = sklearn.metrics.accuracy_score
    else:
        ValueError("inputed random forest's class is not 1 of the expected options")

    pred = random_forest.predict(X_test)
    return scoring(y_test, pred)


##### analysis #####

## data creation ##
X,y = pull_data(data_set, path, reg_or_class)
seed = get_random_seed()

np.random.seed(seed)
X_train, y_train, X_test, y_test = split_data(X,y,
                                              test_size=.5)



## storage prep ##
scoring_dict = dict()
info_dict = dict()

## models ##

# base rf --------------
rf_base = generate_rf(X_train, y_train, n_trees,
                      reg_or_class=reg_or_class)

info_dict["seed"] = seed
scoring_dict["rf_base"] =  assess_rf(rf_base,
                                     X_test, y_test)

random_forest = copy.deepcopy(rf_base)


# depth analysis --------------
print("Depth tune\n")
depth_tune_rf = smooth_rf.depth_tune(random_forest,
                                     X_trained=X_train,
                                     y_trained=y_trained)

scoring_dict["depth_tune"] =  assess_rf(depth_tune_rf,
                                     X_test, y_test)
info_dict["depth_tune"] = depth_tune_rf.loss_vec_depth
# wOLS analysis --------------
print("wOLS")
inner_distance_opts = ["standard", "max", "min"] #inner_distance
parent_all_opts = [True, False] # parent_all
no_constraint_opts = [True, False] # no_constraint

for inner_distance, parent_all, no_constraint_opts in \
     itertools.product(inner_distance_opts, parent_all_opts, no_constraint_opts):
            wOLS_opt_rf = smooth_rf.smooth_all(random_forest,
                                               X_trained=X_train,
                                               y_trained=y_train,
                                               no_constraint=no_constraint,
                                               parents_all=parent_all,
                                               verbose=True)

            name = "wOLs_opt" + "dist:" + inner_distance +\
                        "parents:" + str(parents_all) +\
                        "constraints:" + str(not no_constraint)

            scoring_dict[name] = assess_rf(wOLS_opt_rf,
                                     X_test, y_test)
            info_dict[name] = wOLS_opt_rf.lamb


# saving before ----------
pickle # fill in

# element approach --------------
print("element approach, Adam")

# adam
alphas = [10**-3, 10**-2, 10**-1, .3]
beta_1s = [.1,.3,.5,.9,.999]
beta_0s = [.9,.95,.99]
epsilons = [10**-8, 10**-4]


inner_distance_opts = ["standard", "max", "min"] #inner_distance
parent_all_opts = [True, False] # parent_all
no_constraint_opts = [True, False] # no_constraint
initial_lamb_opts in [seed, None] # initial_lamb
class_loss_opts in ["ce", "l2"] # class_loss
adam_values_opts in itertools.product(alphas, beta_1s, beta_0s, epsilons) # adam_values
for inner_distance, parent_all, no_constraint, initial_lamb, class_loss, \
    adam_values in itertools.product(inner_distance_opts,
                                     parent_all_opts,
                                     no_constraint_opts,
                                     initial_lamb_opts,
                                     class_loss_opts,
                                     adam_values_opts):
    a, b1, b2, e = adam_values
    adam_dict = {"alpha": a,
                 "beta_1": b1,
                 "beta_2": b2,
                 "eps": e}
    adam_rf, _ , _ , c = smooth_rf.smooth(
                            random_forest,
                            X_trained=X_train,
                            y_trained=y_train,
                            no_constraint=no_constraint,
                            sgd_max_num=10000,
                            all_trees=False,
                            initial_lamb_seed=initial_lamb,
                            parents_all=False,
                            distance_style=inner_distance,
                            class_eps=0.0001,
                            class_loss=class_loss,
                            adam=adam_dict)
    if seed is not None:
        il = str(seed)
    else:
        il = "rf"

    name = "element_opt" + "dist:" + inner_distance +\
        "parents:" + str(parents_all) +\
        "constraints:" + str(not no_constraint) +\
        "initial_lamb:" + il +\
        "adam_options:" + str(adam_values).replace(", ", "_")

    scoring_dict[name] = assess_rf(wOLS_opt_rf,
                                   X_test, y_test)
    info_dict[name] = wOLS_opt_rf.lamb


