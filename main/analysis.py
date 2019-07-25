import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn
import progressbar
import sklearn.model_selection
import sklearn.datasets
import pdb
import sys
import copy
import pickle
import itertools
from joblib import Parallel, delayed
import time

import smooth_rf

path = "../"

data_set = sys.argv[1]
reg_or_class = sys.argv[2]
n_trees = np.int(sys.argv[3])

print(data_set)

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
    elif type(random_forest) == sklearn.ensemble.RandomForestClassifier:
        scoring = sklearn.metrics.accuracy_score
    else:
        ValueError("inputed random forest's class is not 1 of the expected options")

    pred = random_forest.predict(X_test)
    return scoring(y_test, pred)


##### analysis #####

## data creation ##
X,y = pull_data(data_set, path, reg_or_class)
my_seed = get_random_seed()

np.random.seed(my_seed)
X_train, y_train, X_test, y_test = split_data(X,y,
                                              test_size=.5)



## storage prep ##
scoring_dict = dict()
info_dict = dict()

## models ##

# base rf --------------
rf_base = generate_rf(X_train, y_train, n_trees,
                      reg_or_class=reg_or_class)

info_dict["seed"] = my_seed
scoring_dict["rf_base"] =  assess_rf(rf_base,
                                     X_test, y_test)

random_forest = copy.deepcopy(rf_base)


# depth analysis --------------
print("Depth tune:")
time_start = time.time()

depth_tune_rf = smooth_rf.depth_tune(random_forest,
                                     X_trained=X_train,
                                     y_trained=y_train)

scoring_dict["depth_tune"] =  assess_rf(depth_tune_rf,
                                     X_test, y_test)
info_dict["depth_tune"] = depth_tune_rf.loss_vec_depth

depth_spent = time.time() - time_start
times = [depth_spent]

# wOLS analysis --------------
if reg_or_class == "reg":
    print("wOLS:")
    time_start = time.time()

    parent_all_opts = [True, False] # parent_all
    no_constraint_opts = [True, False] # no_constraint

    # for parent_all, no_constraint in \
    #      itertools.product(parent_all_opts, no_constraint_opts):
    #             wOLS_opt_rf = smooth_rf.smooth_all(random_forest,
    #                                                X_trained=X_train,
    #                                                y_trained=y_train,
    #                                                no_constraint=no_constraint,
    #                                                parents_all=parent_all,
    #                                                distance
    #                                                verbose=True)

    #             name = "wOLs_opt" +\
    #                         "parents:" + str(parent_all) +\
    #                         "constraints:" + str(not no_constraint)

    #             scoring_dict[name] = assess_rf(wOLS_opt_rf,
    #                                      X_test, y_test)
    #             info_dict[name] = wOLS_opt_rf.lamb



    def smooth_all_wrapper(random_forest,
                       X_train, y_train,
                       params):
        parent_all, no_constraint = params
        wOLS_opt_rf = smooth_rf.smooth_all(random_forest,
                                           X_trained=X_train,
                                           y_trained=y_train,
                                           no_constraint=no_constraint,
                                           parents_all=parent_all,
                                           verbose=False)

        name = "wOLs_opt" +\
                    "_parents:" + str(parent_all) +\
                    ",constraints:" + str(not no_constraint)

        scoring = assess_rf(wOLS_opt_rf,
                                 X_test, y_test)
        info = wOLS_opt_rf.lamb

        return scoring, info, name

    s_all_output = Parallel(n_jobs=-1, verbose=10)(delayed(smooth_all_wrapper)(random_forest,
                                           X_train,
                                           y_train,
                                           params) for params in itertools.product(parent_all_opts, no_constraint_opts))

# a = list()
# for params in itertools.product(parent_all_opts, no_constraint_opts):
#     a.append(smooth_all_wrapper(random_forest,
#                                        X_train,
#                                        y_train,
#                                        params))

    for scoring, info, name in s_all_output:
        scoring_dict[name] = scoring
        info_dict[name] = info

    wols_spent = time.time() - time_start
    times.append(wols_spent)

    # saving before ----------
    with open("data/"+data_set+"_"+reg_or_class+"_"+str(n_trees)+\
                "_"+str(my_seed)+".pkl",
              "wb") as pfile:
        pickle.dump({"seed":my_seed,
                     "time":times,
                     "scoring":scoring_dict,
                     "info":info_dict}, file = pfile)
# element approach --------------
print("element approach, Adam")
time_start = time.time()

# adam
alphas = [10**-3, 10**-2, 10**-1, .3]
beta_1s = [.1,.3,.5,.9,.999]
beta_0s = [.9,.95,.99]
epsilons = [10**-8, 10**-4]


inner_distance_opts = ["standard", "max", "min"] #inner_distance
parent_all_opts = [True, False] # parent_all
if reg_or_class == "reg":
    no_constraint_opts = [True, False] # no_constraint
else:
    no_constraint_opts = [False]
initial_lamb_opts = [my_seed, None] # initial_lamb
if reg_or_class == "class":
    class_loss_opts = ["ce", "l2"] # class_loss
else:
    class_loss_opts = ["l2"]
adam_values_opts = itertools.product(alphas, beta_1s, beta_0s, epsilons) # adam_values
# for inner_distance, parent_all, no_constraint, initial_lamb, class_loss, \
#     adam_values in itertools.product(inner_distance_opts,
#                                      parent_all_opts,
#                                      no_constraint_opts,
#                                      initial_lamb_opts,
#                                      class_loss_opts,
#                                      adam_values_opts):
#     a, b1, b2, e = adam_values
#     adam_dict = {"alpha": a,
#                  "beta_1": b1,
#                  "beta_2": b2,
#                  "eps": e}
#     adam_rf, _ , _ , c = smooth_rf.smooth(
#                             random_forest,
#                             X_trained=X_train,
#                             y_trained=y_train,
#                             no_constraint=no_constraint,
#                             sgd_max_num=10000,
#                             all_trees=False,
#                             initial_lamb_seed=initial_lamb,
#                             parents_all=False,
#                             distance_style=inner_distance,
#                             class_eps=0.0001,
#                             class_loss=class_loss,
#                             adam=adam_dict)
#     if seed is not None:
#         il = str(seed)
#     else:
#         il = "rf"

#     name = "element_opt" +\
#         "parents:" + str(parent_all) +\
#         "constraints:" + str(not no_constraint) +\
#         "dist:" + inner_distance +\
#         "initial_lamb:" + il +\
#         "adam_options:" + str(adam_values).replace(", ", "_")

#     scoring_dict[name] = assess_rf(wOLS_opt_rf,
#                                    X_test, y_test)
#     info_dict[name] = wOLS_opt_rf.lamb

def smooth_wrapper(random_forest,
                   X_train, y_train,
                   params):
    inner_distance, parent_all, no_constraint, initial_lamb, class_loss, \
    adam_values = params

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
        adam=adam_dict,
        verbose=False)

    best_oob = np.min(c)

    if initial_lamb is not None:
        il = str(initial_lamb)
    else:
        il = "rf"

    name = "element_opt" +\
        "_dist:" + inner_distance +\
        ",parents:" + str(parent_all) +\
        ",constraints:" + str(not no_constraint) +\
        ",initial_lamb:" + il +\
        ",adam_options:" + str(adam_values).replace(", ", "_")

    scoring = assess_rf(adam_rf, X_test, y_test)
    info = adam_rf.lamb

    return name, scoring, info, best_oob

num_jobs = len(list(itertools.product(inner_distance_opts,
                                     parent_all_opts,
                                     no_constraint_opts,
                                     initial_lamb_opts,
                                     class_loss_opts,
                                     list(itertools.product(alphas, beta_1s, beta_0s, epsilons)))))

print("must complete " + str(num_jobs) + " number of jobs")

s_all_output = Parallel(n_jobs=-1, verbose=10)(delayed(smooth_wrapper)(random_forest,
                                       X_train,
                                       y_train,
                                       params) for params in list(itertools.product(inner_distance_opts,
                                     parent_all_opts,
                                     no_constraint_opts,
                                     initial_lamb_opts,
                                     class_loss_opts,
                                     list(itertools.product(alphas, beta_1s, beta_0s, epsilons)))))
# a = list()
# for params in itertools.product(inner_distance_opts,
#                                      parent_all_opts,
#                                      no_constraint_opts,
#                                      initial_lamb_opts,
#                                      class_loss_opts,
#                                      list(itertools.product(alphas, beta_1s, beta_0s, epsilons))):
#     print('.')
#     a.append(smooth_wrapper(random_forest,
#                                        X_train,
#                                        y_train,
#                                        params))

for name, scoring, info, best_oob in s_all_output:
    scoring_dict["smooth_element_based,"+name] = c(scoring, best_oob)
    info_dict["smooth_element_based,"+name] = info

adam_spent = time.time() - time_start
times.append(adam_spent)

# saving
with open("data/"+data_set+"_"+reg_or_class+"_"+str(n_trees)+\
            "_"+str(my_seed)+".pkl",
          "wb") as pfile:
    pickle.dump({"seed":my_seed,
                 "time":times,
                 "scoring":scoring_dict,
                 "info":info_dict}, file = pfile)



# element approach --------------
if False:
    print("pytorch approach")
    time_start = time.time()

    inner_distance_opts = ["standard"] #inner_distance (note no option inner_distance_opts checked)
    parent_all_opts = [True, False] # parent_all
    initial_lamb_opts = ["r", 10] # initial_lamb
    which_dicts_opts = [[], ["one_d_dict"], ["two_d_dict"],["one_d_dict", "two_d_dict"]] # which_dicts
    x_dicts_opts = [[], ["one_d_dict"], ["two_d_dict"],["one_d_dict", "two_d_dict"]]
    adam_values_opts = itertools.product(alphas, beta_1s, beta_0s, epsilons) # adam_values

    def smooth_pytorch_wrapper(random_forest,
                       X_train, y_train,
                       params):
        inner_distance, parent_all, initial_lamb, which_dicts, \
            x_dicts, adam_values = params

        a, b1, b2, e = adam_values
        adam_dict = {"lr": a,
                     "betas": (b1, b2),
                     "eps": e}

        if len(which_dicts) > 0 or len(x_dicts) > 0:
            smooth_rf_model, _ , loss_min, params_min, _, _ = smooth_rf.smooth_pytorch(
                random_forest,
                X_trained=X_train,
                y_trained=y_train,
                sgd_max_num=10000,
                all_trees=False,
                init=initial_lamb,
                parents_all=False,
                distance_style=inner_distance,
                adam=adam_dict,
                verbose=False)
        else:
            smooth_rf_model = random_forest
            params_min = "random forest not optimized"
            loss_min = -np.inf

        if initial_lamb == 10:
            il = "rf"
        else:
            il = "r"

        wd = ""
        if "one_d_dict" in which_dicts:
            wd += "1d"
        if "two_d_dict" in which_dicts:
            wd += "2d"

        xd = ""
        if "one_d_dict" in x_dicts:
            xd += "1d"
        if "two_d_dict" in x_dicts:
            xd += "2d"

        name = "element_opt" +\
            "_dist:" + inner_distance +\
            ",parents:" + str(parent_all) +\
            ",initial_lamb:" + il +\
            ",wd:" + wd +\
            ",xd:" + xd +\
            ",adam_options:" + str(adam_values).replace(", ", "_")

        scoring = assess_rf(smooth_rf_model, X_test, y_test)
        info = None

        return name, scoring, info



    num_jobs = len(list(itertools.product(inner_distance_opts,
                                          parent_all_opts,
                                          initial_lamb_opts,
                                          which_dicts_opts,
                                          x_dicts_opts,
                                          list(itertools.product(alphas, beta_1s, beta_0s, epsilons)))))

    print("must complete " + str(num_jobs) + " number of jobs")

    s_all_output = Parallel(n_jobs=-1, verbose=10)(delayed(smooth_pytorch_wrapper)(random_forest,
                                           X_train,
                                           y_train,
                                           params) for params in list(itertools.product(inner_distance_opts,
                                          parent_all_opts,
                                          initial_lamb_opts,
                                          which_dicts_opts,
                                          x_dicts_opts,
                                          list(itertools.product(alphas, beta_1s, beta_0s, epsilons)))))


    a = list()
    for params in itertools.product(inner_distance_opts,
                                          parent_all_opts,
                                          initial_lamb_opts,
                                          which_dicts_opts,
                                          x_dicts_opts,
                                          list(itertools.product(alphas, beta_1s, beta_0s, epsilons))):
        print('.')
        a.append(smooth_pytorch_wrapper(random_forest,
                                           X_train,
                                           y_train,
                                           params))


    for name, scoring, info in s_all_output:
        scoring_dict["smooth_pytorch_based,"+name] = scoring
        #info_dict[name] = info

    pytorch_spent = time.time() - time_start
    times.append(pytorch_spent)

    # saving
    with open("data/"+data_set+"_"+reg_or_class+"_"+str(n_trees)+\
                "_"+str(my_seed)+".pkl",
              "wb") as pfile:
        pickle.dump({"seed":my_seed,
                     "time":times,
                     "scoring":scoring_dict,
                     "info":info_dict}, file = pfile)
