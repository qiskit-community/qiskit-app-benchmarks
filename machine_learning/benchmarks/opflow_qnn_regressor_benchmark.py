# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Opflow based neural network regressor benchmarks."""

import pickle
from itertools import product
from timeit import timeit
from typing import Optional

from qiskit.algorithms.optimizers.cobyla import COBYLA
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base_regressor_benchmark import (
    BaseRegressorBenchmark,
    DATASET_SYNTHETIC_REGRESSION,
    DATASET_CCPP_REGRESSION,
)


class OpflowQnnRegressorBenchmarks(BaseRegressorBenchmark):
    """Opflow QNN regressor benchmarks."""

    version = 1
    timeout = 1200.0
    params = [
        [DATASET_SYNTHETIC_REGRESSION, DATASET_CCPP_REGRESSION],
        ["qasm_simulator", "statevector_simulator"],
    ]
    param_names = ["dataset", "backend name"]

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[NeuralNetworkRegressor] = None
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

    def setup_cache(self) -> None:
        """Cache Opflow fitted model."""
        for dataset, backend in product(*self.params):
            train_features = self.datasets[dataset]["train_features"]
            train_labels = self.datasets[dataset]["train_labels"]

            if dataset == DATASET_SYNTHETIC_REGRESSION:
                model = self._construct_qnn_synthetic(
                    quantum_instance_name=backend, optimizer=COBYLA()
                )
            elif dataset == DATASET_CCPP_REGRESSION:
                model = self._construct_qnn_ccpp(
                    quantum_instance_name=backend, optimizer=COBYLA(maxiter=100)
                )
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            model.fit(train_features, train_labels)

            file_name = f"{dataset}_{backend}.pickle"
            with open(file_name, "wb") as file:
                pickle.dump(model._fit_result, file)

    def setup(self, dataset: str, quantum_instance_name: str) -> None:
        """Set up the benchmark."""

        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]
        self.test_features = self.datasets[dataset]["test_features"]
        self.test_labels = self.datasets[dataset]["test_labels"]

        if dataset == DATASET_SYNTHETIC_REGRESSION:
            self.model = self._construct_qnn_synthetic(quantum_instance_name=quantum_instance_name)
        elif dataset == DATASET_CCPP_REGRESSION:
            self.model = self._construct_qnn_ccpp(quantum_instance_name=quantum_instance_name)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        file_name = f"{dataset}_{quantum_instance_name}.pickle"
        with open(file_name, "rb") as file:
            self.model._fit_result = pickle.load(file)

    # pylint: disable=invalid-name
    def time_score_opflow_qnn_regressor(self, _, __):
        """Time scoring OpflowQNN regressor on data."""
        self.model.score(self.train_features, self.train_labels)

    def time_predict_opflow_qnn_regressor(self, _, __):
        """Time predicting with OpflowQNN regressor."""
        self.model.predict(self.train_features)

    def track_score_opflow_qnn_regressor(self, _, __):
        """R2 score of the model on data."""
        return self.model.score(self.test_features, self.test_labels)

    def track_mae_opflow_qnn_regressor(self, _, __):
        """Mean absolute error of the model on data."""
        predicts = self.model.predict(self.test_features)
        mae = mean_absolute_error(y_true=self.test_labels, y_pred=predicts)
        return mae

    def track_mse_opflow_qnn_regressor(self, _, __):
        """Mean squared error of the model on data."""
        predicts = self.model.predict(self.test_features)
        mse = mean_squared_error(y_true=self.test_labels, y_pred=predicts)
        return mse


if __name__ == "__main__":
    bench = OpflowQnnRegressorBenchmarks()
    bench.setup_cache()
    for dataset_name, backend_name in product(*OpflowQnnRegressorBenchmarks.params):
        try:
            bench.setup(dataset_name, backend_name)
        except NotImplementedError:
            continue

        for method in (
            "time_score_opflow_qnn_regressor",
            "time_predict_opflow_qnn_regressor",
            "track_score_opflow_qnn_regressor",
            "track_mae_opflow_qnn_regressor",
            "track_mse_opflow_qnn_regressor",
        ):
            elapsed = timeit(
                f'bench.{method}("{dataset_name}", "{backend_name}")', number=10, globals=globals()
            )
            print(f"{method}:\t{elapsed}")
