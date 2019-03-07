import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import progressbar
import copy
import sklearn.ensemble
import sklearn
import matplotlib.path as mpltPath
import pdb


def generate_data(large_n = 650):
    """
    generate data structure similar to example on page 114 of [Criminisi 2012]

    Arguments:
    ----------
    large_n : integer
      approximately the number of observations in the dataset

    Returns:
    --------
    data : array (~large_n, 2)
      array with 2d features space for the observations
    y : array (~large_n,)
      1d vector with class structure of observations

    Note:
    -----
    the quanity of observataions will be approximately large_n

    Citations:
    ----------
    [Criminisi 2012]
    Criminisi A, Shotton J, Konukoglu E. Decision forests: A unified framework
    for classification, regression, density estimation, manifold learning and
    semi-supervised learning. Foundations and Trends® in Computer Graphics and
    Vision. 2012 Mar 29;7(2–3):81-227.
    """
    n = np.int(large_n / (4 + 1/3))
    unif_prob = .6
    norm_prob = 1 - unif_prob
    rotate_angle = -10
    delta = .05

    d1_n = np.random.normal(size = (np.int(n * norm_prob),2),
                            loc = [.5,.9],
                            scale = [.33,.05])
    d1_u = np.vstack( ( np.random.uniform(size = np.int(n * unif_prob)),
                        np.random.uniform(size = np.int(n * unif_prob) ,
                                          low = .8)) ).T


    d2_n = np.random.normal(size = (np.int(n * norm_prob),2),
                            loc = [.9,.5],
                            scale = [.05,.33])
    d2_u = np.vstack( ( np.random.uniform(size = np.int(n * unif_prob), low = .8),
                        np.random.uniform(size = np.int(n * unif_prob))) ).T

    d3_u_pre = np.random.uniform(size = (n,2), high = .5)
    d3_u_rotate = d3_u_pre @ \
                    np.array([[np.cos(rotate_angle * (np.pi/180)),
                                    -np.sin(rotate_angle * (np.pi/180))],
                              [np.sin(rotate_angle * (np.pi/180)),
                                    np.cos(rotate_angle * (np.pi/180))]])

    d3_u = d3_u_rotate + np.array([.175,.075])


    d4_1_n = np.random.normal(size = (np.int(n * 2/3 * norm_prob),2),
                              loc = [.5,.1],
                              scale = [.33,.05])
    d4_1_u = np.vstack( ( np.random.uniform(size = np.int(n * 2/3 * unif_prob)),
                        np.random.uniform(size = np.int(n * 2/3 * unif_prob),
                                          high = .2)) ).T

    d4_2_n = np.random.normal(size = (np.int(n * norm_prob),2),
                              loc = [.1,.5], scale = [.05,.33])
    d4_2_u = np.vstack( ( np.random.uniform(size = np.int(n * 2/3 * unif_prob),
                                            high = .2),
                        np.random.uniform(size = np.int(n * 2/3 * unif_prob) )) ).T


    data = np.vstack((d1_n, d1_u,
                      d2_n, d2_u,
                      d3_u,
                      d4_1_n, d4_1_u, d4_2_n, d4_2_u))

    n_square = np.sum([x.shape[0] for x in [d4_1_n, d4_1_u, d4_2_n, d4_2_u]],
                      dtype = np.int)


    y = np.array([2]*(d1_n.shape[0] + d1_u.shape[0]) +\
                 [3]*(d2_n.shape[0] + d2_u.shape[0]) +\
                 [0]*(d3_u.shape[0]) + [1]*n_square, dtype = np.int)

    # removing points:
    rm_logic = (data[:,0] > 1 + delta) + (data[:,0] < 0 - delta) +\
               (data[:,1] > 1 + delta) + (data[:,1] < 0 - delta)
    keep_logic = rm_logic == 0

    data = data[keep_logic,:]
    y = y[keep_logic]

    return data, y

def generate_data_knn(n = 650, p = np.array([.4,.6])):
  """
  generate 2d example that can be overfit by knn (classes cross border)

  Arguments:
  ----------
  n : integer
    number of observations
  p : array (2,)
    vector of proportions of classes

  Returns:
  --------
  new : array (n, 2)
    array with 2d features space for the observations
  y : array (n,)
    1d vector with class structure of observations
  value : array (n, )
    number of constraining lines the observation is above/ to the left off
  """

  # data structure:

  data = pd.DataFrame(
    data = {"x" :
    [0.008754602, 0.013429834, 0.027455531, 0.050831692, 0.122518587,
    0.187971838, 0.230048928, 0.261217143, 0.259658733, 0.251866679,
    0.279918072, 0.317319930, 0.353163378, 0.378097950, 0.374981128,
    0.376539539, 0.409266165, 0.440434380, 0.463810541, 0.487186702,
    0.535497435, 0.549523132, 0.583808169, 0.593158633, 0.593158633,
    0.588483401, 0.561990418, 0.540172668, 0.521471739, 0.526146971,
    0.588483401, 0.600950687, 0.600950687, 0.608742741, 0.728740368,
    0.748999708, 0.759908583, 0.775492691, 0.792635209, 0.806660906,
    0.823803424, 0.845621175, 0.845621175, 0.872114157, 0.900165551,
    0.915749658, 0.915749658, 0.921983301, 0.931333766, 0.939125820,
    0.962501981, 0.985878142],
    "y":
    [0.04020202, 0.08815571, 0.10680437, 0.12545303, 0.16008626,
    0.11213256, 0.09348390, 0.09348390, 0.16008626, 0.20537586,
    0.21603224, 0.18672720, 0.14676579, 0.14676579, 0.20271177,
    0.25599365, 0.27997050, 0.27997050, 0.24267318, 0.22402452,
    0.25066547, 0.27997050, 0.37587790, 0.41051112, 0.48244167,
    0.52773127, 0.57568497, 0.62363867, 0.65560780, 0.72221016,
    0.74352291, 0.75417929, 0.81545346, 0.85008669, 0.83676621,
    0.80746118, 0.77282795, 0.70089740, 0.63695914, 0.54904403,
    0.52773127, 0.51174671, 0.51174671, 0.51174671, 0.51973899,
    0.51973899, 0.49043395, 0.40784703, 0.37054971, 0.35190105,
    0.33325239, 0.31193963]}, columns = ["x","y"])


  upper = pd.DataFrame(
    data = {"x" :
    [-0.013152278,  0.008168012,  0.052331469,  0.096494926,  0.136089750,
    0.174161696,  0.210710764,  0.242691198, 0.235076809,  0.235076809,
    0.268580121,  0.302083434,  0.326449479,  0.346246891,  0.359952791,
    0.367567180, 0.373658692,  0.402593371,  0.433050927,  0.457416973,
    0.472645751,  0.492443163,  0.519854964,  0.539652376, 0.560972665,
    0.571632810,  0.570109932,  0.562495543,  0.545743887,  0.528992231,
    0.510717697,  0.501580430, 0.501580430,  0.500057552,  0.519854964,
    0.541175253,  0.559449787,  0.574678566,  0.772652684,  0.793972974,
    0.809201752,  0.818339019,  0.827476286,  0.847273698,  0.874685499,
    0.903620178,  0.924940468,  0.941692124, 0.947783635,  0.950829391,
    0.964535291,  0.979764070,  0.987378459],
    "y":
    [0.0542169, 0.1300291, 0.1695833, 0.1893605, 0.1893605, 0.1662872,
    0.1465101, 0.1366215, 0.1959528, 0.2256185, 0.2420994, 0.2388032,
    0.2223223, 0.2058414, 0.2025452, 0.2552841, 0.2981345, 0.3113193,
    0.3212078, 0.3047269, 0.2915422, 0.2849498, 0.3014307, 0.3409849,
    0.3871315, 0.4266857, 0.4794247, 0.5024980, 0.5288674, 0.5618293,
    0.5914949, 0.6244568, 0.6837881, 0.7365270, 0.7793774, 0.8123393,
    0.8420049, 0.8650782, 0.8749668, 0.8123393, 0.7497118, 0.6936766,
    0.6574186, 0.6277530, 0.6079759, 0.6112720, 0.6112720, 0.5684216,
    0.5387560, 0.5024980, 0.4530552, 0.4233895, 0.4102048]},
    columns = ["x","y"])

  lower = pd.DataFrame(
    data = {"x" :
    [0.04319420, 0.07212888, 0.11324658, 0.14370414, 0.18177608,
    0.21984803, 0.26705724, 0.29751480, 0.29903768, 0.32340372,
    0.35690704, 0.40107049, 0.40411625, 0.41934503, 0.43761956,
    0.46198561, 0.48787453, 0.54269813, 0.58381583, 0.60209037,
    0.62950217, 0.62493353, 0.61427339, 0.58533871, 0.56097267,
    0.55640403, 0.59447598, 0.62188778, 0.62188778, 0.66148260,
    0.70412318, 0.73610362, 0.75285527, 0.76351542, 0.77417556,
    0.79397297, 0.83509068, 0.86097960, 0.88382277, 0.89752867,
    0.91123457, 0.94169212, 0.97976407],
    "y":
    [0.03443980, 0.06740164, 0.09047493, 0.07399401, 0.05092072,
    0.04432835, 0.04432835, 0.08388256, 0.12673295, 0.11684440,
    0.10695585, 0.11025203, 0.15639861, 0.21243374, 0.20584138,
    0.19595282, 0.18606427, 0.20584138, 0.26517269, 0.32120782,
    0.41350098, 0.46953611, 0.52886743, 0.58490256, 0.63434532,
    0.67060335, 0.69038046, 0.71674993, 0.77608125, 0.79915453,
    0.81233927, 0.76948888, 0.68708427, 0.58819875, 0.48931322,
    0.44975901, 0.42009335, 0.42009335, 0.41350098, 0.35416967,
    0.31791164, 0.27176506, 0.24869177]}, columns = ["x","y"])

  left_add = np.min([lower.x.min(),
                     upper.x.min(),
                     data.x.min()])

  right_add = np.max([lower.x.max(),
                     upper.x.max(),
                     data.x.max()])

  above = upper.y.max() + .1
  below = lower.y.min() - .1



  data_np = np.concatenate((np.array([[left_add,data.y[0]]]),
                            np.array(data),
                            np.array([[right_add,data.y[len(data.y) -1]]]),
                            np.array([[right_add, above],
                                      [left_add, above]]))
                            )
  lower_np = np.concatenate((np.array([[left_add,lower.y[0]]]),
                            np.array(lower),
                            np.array([[right_add,lower.y[len(lower.y) -1]]]),
                            np.array([[right_add, above],
                                      [left_add, above]]))
                            )
  upper_np = np.concatenate((np.array([[left_add,upper.y[0]]]),
                            np.array(upper),
                            np.array([[right_add,upper.y[len(upper.y) -1]]]),
                            np.array([[right_add, above],
                                      [left_add, above]]))
                            )

  # actual creation:
  new = np.hstack((np.random.uniform(low = left_add,
                                     high = right_add,
                                     size = n).reshape((-1,1)),
                   np.random.uniform(low = below,
                                     high = above,
                                     size = n).reshape((-1,1))))

  path_l = mpltPath.Path(lower_np)
  path_c = mpltPath.Path(data_np)
  path_u = mpltPath.Path(upper_np)

  inside_l = path_l.contains_points(new)
  inside_c = path_c.contains_points(new)
  inside_u = path_u.contains_points(new)

  value = inside_l * 1 + \
          inside_c * 1 + \
          inside_u * 1

  y = np.zeros(new.shape[0])
  for v in np.arange(1,4, dtype = np.int):
    if v == 1:
      y[value == v] = np.random.binomial(n = 1,
                                         size = np.sum(value == v),
                                         p = p[0])
    if v == 2:
      y[value == v] = np.random.binomial(n = 1,
                                         size = np.sum(value == v),
                                         p = p[1])
    if v == 3:
      y[value == v] = 1

  # plt.plot(data_np[:,0], data_np[:,1])
  # plt.plot(lower_np[:,0], lower_np[:,1])
  # plt.plot(upper_np[:,0], upper_np[:,1])
  # plt.scatter(new[:,0], new[:,1],
  #             c = np.array(["r","b"])[(y).astype(np.int)])

  return new, y, value


#### generating spiral function

def spirals(n_total = 600, n_classes = 3, noise_sd = .1, t_shift = 0):
    """
    create spirals dataset

    Arguments:
    ----------
    n_total: int
        total number of observations (up to rounding - all classes get the
        same number of observations)
    n_classes: int
        number of classes
    noise_sd: scalar
        standard deviation of student t noise added in (x,y) dimension
    t_shift: scalar
        shifts start of t from 0 to t_shift
        (so that classes don't connect at center)

    Returns
    -------
    data_all: pd dataframe, with columns: t, x, y, class (class is the label)
    """
    shifts = np.linspace(0, 2*np.pi, num = n_classes, endpoint = False)
    n_each = np.int(n_total/n_classes)


    data_all = pd.DataFrame()

    for c_idx, shift in enumerate(shifts):
        tt =  np.random.uniform(low = 0, high = 2*np.pi, size = n_each) +\
            shift + t_shift
        #np.linspace(0, 2*np.pi, num = n_each) + shift + t_shift
        inner_data = pd.DataFrame(
                        data = {"t": tt,
                                "x": np.sqrt(tt - shift) * np.cos(tt) +\
                                    noise_sd *\
                                        np.random.standard_t(df = 1,
                                                             size = n_each),
                                "y": np.sqrt(tt - shift) * np.sin(tt) +\
                                    noise_sd *\
                                        np.random.standard_t(df = 1,
                                                             size = n_each),
                                "class": c_idx
                                })

        data_all = data_all.append(inner_data)

    return data_all
