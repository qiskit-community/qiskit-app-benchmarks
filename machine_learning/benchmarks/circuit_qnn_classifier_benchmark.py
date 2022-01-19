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
"""Circuit QNN Classifier benchmarks."""

import pickle
from itertools import product
from timeit import timeit
from typing import Optional

from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from .circuit_qnn_base_classifier_benchmark import CircuitQnnBaseClassifierBenchmark
from .base_classifier_benchmark import (
    DATASET_SYNTHETIC_CLASSIFICATION,
    DATASET_IRIS_CLASSIFICATION,
)


class CircuitQnnClassifierBenchmarks(CircuitQnnBaseClassifierBenchmark):
    """Circuit QNN Classifier benchmarks."""

    version = 2
    timeout = 1200.0
    params = [
        [DATASET_SYNTHETIC_CLASSIFICATION, DATASET_IRIS_CLASSIFICATION],
        ["qasm_simulator", "statevector_simulator"],
    ]
    param_names = ["dataset", "backend name"]

    def __init__(self) -> None:
        super().__init__()
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.model: Optional[NeuralNetworkClassifier] = None

    def setup(self, dataset: str, quantum_instance_name: str) -> None:
        """Set up the benchmark."""

        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]
        self.test_features = self.datasets[dataset]["test_features"]
        self.test_labels = self.datasets[dataset]["test_labels"]

        if dataset == DATASET_SYNTHETIC_CLASSIFICATION:
            self.model = self._construct_qnn_classifier_synthetic(
                quantum_instance_name=quantum_instance_name
            )
        elif dataset == DATASET_IRIS_CLASSIFICATION:
            self.model = self._construct_qnn_classifier_iris(
                quantum_instance_name=quantum_instance_name
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        file_name = f"circuit_qnn_{dataset}_{quantum_instance_name}.pickle"
        with open(file_name, "rb") as file:
            self.model._fit_result = pickle.load(file)

    def setup_cache(self) -> None:
        """Cache CircuitQNN fitted model"""
        for dataset, backend in product(*self.params):
            train_features = self.datasets[dataset]["train_features"]
            train_labels = self.datasets[dataset]["train_labels"]

            if dataset == DATASET_SYNTHETIC_CLASSIFICATION:
                model = self._construct_qnn_classifier_synthetic(
                    quantum_instance_name=backend, optimizer=COBYLA(maxiter=200)
                )
            elif dataset == DATASET_IRIS_CLASSIFICATION:
                model = self._construct_qnn_classifier_iris(
                    quantum_instance_name=backend, optimizer=COBYLA(maxiter=200)
                )
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            model.fit(train_features, train_labels)

            file_name = f"circuit_qnn_{dataset}_{backend}.pickle"
            with open(file_name, "wb") as file:
                pickle.dump(model._fit_result, file)

    # pylint: disable=invalid-name
    def time_score_circuit_qnn_classifier(self, _, __):
        """Time scoring CircuitQNN classifier on data."""
        self.model.score(self.train_features, self.train_labels)

    def time_predict_circuit_qnn_classifier(self, _, __):
        """Time predicting with CircuitQNN classifier."""
        self.model.predict(self.train_features)

    def track_accuracy_score_circuit_qnn_classifier(self, _, __):
        """Tracks the overall accuracy of the classification results."""
        return self.model.score(self.test_features, self.test_labels)

    def track_precision_score_circuit_qnn_classifier(self, _, __):
        """Tracks the precision score."""
        predicts = self.model.predict(self.test_features)
        return precision_score(y_true=self.test_labels, y_pred=predicts, average="micro")

    def track_recall_score_circuit_qnn_classifier(self, _, __):
        """Tracks the recall score for each class of the classification results."""
        predicts = self.model.predict(self.test_features)
        return recall_score(y_true=self.test_labels, y_pred=predicts, average="micro")

    def track_f1_score_circuit_qnn_classifier(self, _, __):
        """Tracks the f1 score for each class of the classification results."""
        predicts = self.model.predict(self.test_features)
        return f1_score(y_true=self.test_labels, y_pred=predicts, average="micro")


if __name__ == "__main__":
    bench = CircuitQnnClassifierBenchmarks()
    bench.setup_cache()
    for dataset_name, backend_name in product(*CircuitQnnClassifierBenchmarks.params):
        try:
            bench.setup(dataset_name, backend_name)
        except NotImplementedError:
            continue

        for method in (
            "time_score_circuit_qnn_classifier",
            "time_predict_circuit_qnn_classifier",
            "track_accuracy_score_circuit_qnn_classifier",
            "track_precision_score_circuit_qnn_classifier",
            "track_recall_score_circuit_qnn_classifier",
            "track_f1_score_circuit_qnn_classifier",
        ):
            elapsed = timeit(
                f'bench.{method}("{dataset_name}", "{backend_name}")', number=10, globals=globals()
            )
            print(f"{method}:\t{elapsed}")
