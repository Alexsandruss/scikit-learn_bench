# ===============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import io
import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (  # accuracy metrics; regression metrics; clustering metrics
    accuracy_score,
    balanced_accuracy_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ..datasets import load_data
from ..datasets.transformer import split_and_transform_data
from ..utils.bench_case import get_bench_case_value, get_data_name
from ..utils.common import custom_format, get_module_members
from ..utils.config import bench_case_filter
from ..utils.custom_types import BenchCase, Numeric, NumpyNumeric
from ..utils.logger import logger
from ..utils.measurement import measure_case
from ..utils.special_params import assign_case_special_values_on_run
from .common import main_template
from .sklearn_estimator import estimator_to_task, get_number_of_classes, get_estimator
import dpctl
import dpctl.tensor as dpt
from mpi4py import MPI


def get_estimator_methods(bench_case: BenchCase) -> Dict[str, List[str]]:
    # default estimator methods
    estimator_methods = get_bench_case_value(bench_case, f"algorithm:estimator_methods", {"training": "fit", "inference": "predict"})
    for stage in estimator_methods.keys():
        estimator_methods[stage] = estimator_methods[stage].split("|")
    return estimator_methods


def get_subset_metrics_of_estimator(
    task, stage, estimator_instance, x, y
) -> Dict[str, float]:
    metrics = dict()
    if stage == "training":
        if hasattr(estimator_instance, "n_iter_"):
            iterations = estimator_instance.n_iter_
            if isinstance(iterations, Union[Numeric, NumpyNumeric].__args__):
                metrics.update({"iterations": int(iterations)})
            elif (
                hasattr(iterations, "__len__")
                and len(iterations) == 1
                and isinstance(iterations[0], Union[Numeric, NumpyNumeric].__args__)
            ):
                metrics.update({"iterations": int(iterations[0])})
    if task == "classification":
        y_pred = estimator_instance.predict(x)
        metrics.update(
            {
                "accuracy": float(accuracy_score(dpt.to_numpy(y), dpt.to_numpy(y_pred))),
                "balanced accuracy": float(balanced_accuracy_score(dpt.to_numpy(y), dpt.to_numpy(y_pred))),
            }
        )
    elif task == "regression":
        y_pred = estimator_instance.predict(x)
        metrics.update(
            {
                "RMSE": float(mean_squared_error(dpt.to_numpy(y), dpt.to_numpy(y_pred)) ** 0.5),
                "R2": float(r2_score(dpt.to_numpy(y), dpt.to_numpy(y_pred))),
            }
        )
    elif task == "decomposition":
        if "PCA" in str(estimator_instance) and hasattr(estimator_instance, "score"):
            metrics.update({"average log-likelihood": float(estimator_instance.score(x))})
            if stage == "training":
                metrics.update(
                    {
                        "1st component variance ratio": float(
                            estimator_instance.explained_variance_ratio_[0]
                        )
                    }
                )
    elif task == "clustering":
        if hasattr(estimator_instance, "inertia_"):
            # compute inertia manually using distances to cluster centers
            # provided by KMeans.transform
            metrics.update(
                {
                    "inertia": float(
                        np.power(estimator_instance.transform(x).min(axis=1), 2).sum()
                    )
                }
            )
        if hasattr(estimator_instance, "predict"):
            y_pred = estimator_instance.predict(x)
            metrics.update(
                {
                    "Davies-Bouldin score": float(davies_bouldin_score(x, y_pred)),
                    "homogeneity": float(homogeneity_score(y, y_pred)),
                    "completeness": float(completeness_score(y, y_pred)),
                }
            )
        if "DBSCAN" in str(estimator_instance) and stage == "training" and y is not None:
            clusters = len(
                np.unique(estimator_instance.labels_[estimator_instance.labels_ != -1])
            )
            metrics.update({"clusters": clusters})
            if clusters > 1:
                metrics.update(
                    {
                        "Davies-Bouldin score": float(
                            davies_bouldin_score(x, estimator_instance.labels_)
                        )
                    }
                )
            if len(np.unique(y)) < 128:
                metrics.update(
                    {
                        "homogeneity": float(
                            homogeneity_score(y, estimator_instance.labels_)
                        )
                        if clusters > 1
                        else 0,
                        "completeness": float(
                            completeness_score(y, estimator_instance.labels_)
                        )
                        if clusters > 1
                        else 0,
                    }
                )
    return metrics


def measure_sklearn_estimator(
    bench_case,
    task,
    estimator_class,
    estimator_methods,
    estimator_params,
    x_train,
    x_test,
    y_train,
    y_test,
):
    if y_train is not None:
        data_args = {"training": (x_train, y_train), "inference": (x_test,)}
    else:
        data_args = {"training": (x_train,), "inference": (x_test,)}

    metrics = dict()
    estimator_instance = estimator_class(**estimator_params)
    for stage in estimator_methods.keys():
        for method in estimator_methods[stage]:
            if hasattr(estimator_instance, method):
                method_instance = getattr(estimator_instance, method)
                metrics[method] = dict()
                (
                    metrics[method]["time[ms]"],
                    metrics[method]["time std[ms]"],
                    _,
                ) = measure_case(bench_case, method_instance, *data_args[stage])  

    quality_metrics = {
        "training": get_subset_metrics_of_estimator(
            task, "training", estimator_instance, x_train, y_train
        ),
        "inference": get_subset_metrics_of_estimator(
            task, "inference", estimator_instance, x_test, y_test
        ),
    }
    for method in metrics.keys():
        for stage in estimator_methods.keys():
            if method in estimator_methods[stage]:
                metrics[method].update(quality_metrics[stage])

    return metrics, estimator_instance


def main(bench_case: BenchCase, filters: List[BenchCase]):
    # get estimator class and ML task
    library_name = get_bench_case_value(bench_case, "algorithm:library")
    estimator_name = get_bench_case_value(bench_case, "algorithm:estimator")

    estimator_class = get_estimator(library_name, estimator_name, distributed=True)
    task = estimator_to_task(estimator_name)

    if dpctl.has_gpu_devices:
        q = dpctl.SyclQueue("gpu")
    else:
        raise RuntimeError(
            "GPU devices unavailable. Currently, "
            "SPMD execution mode is implemented only for this device type."
        )

    # load and transform data
    data, data_description = load_data(bench_case)
    (x_train, x_test, y_train, y_test), data_description = split_and_transform_data(
        bench_case, data, data_description
    )
    
    train_rows = x_train.shape[0]
    test_rows = x_test.shape[0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    train_start = rank * train_rows // size
    train_end = (1 + rank) * train_rows // size
    test_start = rank * test_rows // size
    test_end = (1 + rank) * test_rows // size

    x_train = dpt.asarray(x_train[train_start:train_end], usm_type="device", sycl_queue=q)
    x_test = dpt.asarray(x_test[test_start:test_end], usm_type="device", sycl_queue=q)
    if y_train is not None:
        y_train = dpt.asarray(y_train[train_start:train_end], usm_type="device", sycl_queue=q)
        y_test = dpt.asarray(y_test[test_start:test_end], usm_type="device", sycl_queue=q)

    # assign special values
    assign_case_special_values_on_run(
        bench_case, x_train, y_train, x_test, y_test, data_description
    )

    # get estimator parameters
    estimator_params = get_bench_case_value(
        bench_case, "algorithm:estimator_params", dict()
    )

    # get estimator methods for measurement
    estimator_methods = get_estimator_methods(bench_case)

    # benchmark case filtering
    if not bench_case_filter(bench_case, filters):
        logger.warning("Benchmarking case was filtered.")
        return list()

    # run estimator methods
    metrics, estimator_instance = measure_sklearn_estimator(
        bench_case,
        task,
        estimator_class,
        estimator_methods,
        estimator_params,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    result_template = {
        "task": task,
        "dataset": get_data_name(bench_case, shortened=True),
        "library": library_name,
        "estimator": estimator_name,
        "device": get_bench_case_value(bench_case, "algorithm:device"),
    }
    # TODO: replace get_params with algorithm estimator_params

    data_descs = {
        "training": data_description["x_train"],
        "inference": data_description["x_test"],
    }
    if "n_classes" in data_description:
        data_descs["training"].update({"n_classes": data_description["n_classes"]})
        data_descs["inference"].update({"n_classes": data_description["n_classes"]})

    results = list()
    for method in metrics.keys():
        result = result_template.copy()
        for stage in estimator_methods.keys():
            if method in estimator_methods[stage]:
                result.update({"stage": stage, "method": method})
                result.update(data_descs[stage])
                result.update(metrics[method])
        results.append(result)
    return results


if __name__ == "__main__":
    main_template(main)
