# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Neural Network Classifier benchmarks."""
from itertools import product
from timeit import timeit

import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZFeatureMap
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class OpflowQnnClassifierBenchmarks(BaseClassifierBenchmark):
    """Opflow QNN Classifier benchmarks."""

    version = 1
    timeout = 1200.0
    params = [
        ["dataset_synthetic", "dataset_iris"],
        ["qasm_simulator", "statevector_simulator"],
    ]
    param_names = ["dataset", "backend name"]

    def setup_dataset_synthetic_classification(self, X, y, quantum_instance_name):
        """Training TwoLayerQNN for synthetic classification dataset."""

        num_inputs = len(X[0])
        self.y = 2 * y - 1  # in {-1, +1}

        feature_map = ZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)

        opflow_qnn = TwoLayerQNN(
            num_inputs,
            feature_map=feature_map,
            ansatz=ansatz,
            quantum_instance=self.backends[quantum_instance_name],
        )

        self.opflow_classifier_fitted = NeuralNetworkClassifier(opflow_qnn, optimizer=COBYLA())
        self.opflow_classifier_fitted.fit(self.X, self.y)

    def setup_dataset_iris(self, X, y, quantum_instance_name):
        """Training TwoLayerQNN for iris classification dataset."""

        num_inputs = len(X[0])

        # keeping only two classes as TwoLayerQNN only supports binary classification
        idx_binary_class = np.where(y != 2)[0]
        self.X = X[idx_binary_class]
        self.y = y[idx_binary_class]

        feature_map = ZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)

        opflow_qnn = TwoLayerQNN(
            num_inputs,
            feature_map=feature_map,
            ansatz=ansatz,
            quantum_instance=self.backends[quantum_instance_name],
        )

        self.opflow_classifier_fitted = NeuralNetworkClassifier(opflow_qnn, optimizer=COBYLA())
        self.opflow_classifier_fitted.fit(self.X, self.y)

    def setup(self, dataset, quantum_instance_name):
        """setup"""

        self.X = self.datasets[dataset]["features"]
        self.y = self.datasets[dataset]["labels"]

        if dataset == "dataset_synthetic":
            self.setup_dataset_synthetic_classification(self.X, self.y, quantum_instance_name)
        elif dataset == "dataset_iris":
            self.setup_dataset_iris(self.X, self.y, quantum_instance_name)

    def time_score_opflow_qnn_classifier(self, _, __):
        """Time scoring OpflowQNN classifier on data."""

        self.opflow_classifier_fitted.score(self.X, self.y)

    def time_predict_opflow_qnn_classifier(self, _, __):
        """Time predicting with classifier OpflowQNN."""

        y_predict = self.opflow_classifier_fitted.predict(self.X)
        return y_predict


if __name__ == "__main__":
    for dataset, backend in product(*OpflowQnnClassifierBenchmarks.params):
        bench = OpflowQnnClassifierBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue

        for method in (
            "time_score_opflow_qnn_classifier",
            "time_predict_opflow_qnn_classifier",
        ):
            elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
