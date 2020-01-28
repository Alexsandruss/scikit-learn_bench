# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, load_data, gen_basic_dict, output_csv
)
import numpy as np
from cuml.svm import SVC
from sklearn.metrics import accuracy_score


def get_optimal_cache_size(n_features, dtype=np.double, max_cache=64):
    '''
    Get an optimal cache size for sklearn.svm.SVC.

    Parameters
    ----------
    n_features : int
        Number of features in the dataset
    dtype : dtype-like, optional (default np.double)
        dtype to use for computing cache size
    max_cache : int, optional (default 64)
        Maximum cache size, in gigabytes
    '''

    byte_size = np.empty(0, dtype=dtype).itemsize
    optimal_cache_size_bytes = byte_size * (n_features ** 2)
    one_gb = 2 ** 30
    max_cache_bytes = max_cache * one_gb
    if optimal_cache_size_bytes > max_cache_bytes:
        return max_cache_bytes
    else:
        return optimal_cache_size_bytes


parser = argparse.ArgumentParser(description='scikit-learn SVM benchmark')

parser.add_argument('-C', dest='C', type=float, default=0.01,
                    help='SVM slack parameter')
parser.add_argument('--kernel', choices=('linear', 'rbf'),
                    default='linear', help='SVM kernel function')
parser.add_argument('--gamma', type=float, default=None,
                    help="Parameter for kernel='rbf'")
parser.add_argument('--maxiter', type=int, default=2000,
                    help='Maximum iterations for the iterative solver. '
                         '-1 means no limit.')
parser.add_argument('--max-cache-size', type=int, default=64,
                    help='Maximum cache size, in gigabytes, for SVM.')
parser.add_argument('--tol', type=float, default=1e-16,
                    help='Tolerance passed to sklearn.svm.SVC')
params = parse_args(parser, loop_types=('fit', 'predict'))

# Load data
X_train, X_test, y_train, y_test = load_data(params)

if params.gamma is None:
    params.gamma = 'auto'

cache_size_bytes = get_optimal_cache_size(X_train.shape[0],
                                          max_cache=params.max_cache_size)
params.cache_size_mb = cache_size_bytes / 1024**2
params.n_classes = len(np.unique(y_train))

# Create our C-SVM classifier
clf = SVC(C=params.C, kernel=params.kernel, max_iter=params.maxiter,
          cache_size=params.cache_size_mb, tol=params.tol,
          gamma=params.gamma)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'kernel', 'cache_size_mb', 'C', 'sv_len', 'n_classes', 'accuracy',
           'time')

# Time fit and predict
fit_time, _ = time_mean_min(clf.fit, X_train, y_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
params.sv_len = clf.support_.shape[0]
y_pred = clf.predict(X_train)
train_acc = 100 * accuracy_score(y_pred, y_train)

predict_time, y_pred = time_mean_min(clf.predict, X_test,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
test_acc = 100 * accuracy_score(y_pred, y_test)

if params.output_format == "csv":
    output_csv(columns, params, functions=['SVM.fit', 'SVM.predict'],
               times=[fit_time, predict_time], accuracies=[None, test_acc])
elif params.output_format == "json":
    import json

    result = gen_basic_dict(
        "cuml", "svc", "training", params, X_train, clf)
    result["input_data"].update({"classes": params.n_classes})
    result.update({
        "time[s]": fit_time,
        "accuracy[%]": train_acc
    })
    print(json.dumps(result, indent=4))

    result = gen_basic_dict(
        "cuml", "svc", "prediction", params, X_test, clf)
    result["input_data"].update({"classes": params.n_classes})
    result.update({
        "time[s]": predict_time,
        "accuracy[%]": test_acc
    })
    print(json.dumps(result, indent=4))