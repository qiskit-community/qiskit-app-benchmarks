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

"""Opflow QNN score benchmarks."""
from itertools import product

import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class OpflowQnnClassifierBenchmarks(BaseClassifierBenchmark):
    """Opflow QNN Score benchmarks."""

    version = 1
    timeout = 1200.0
    params = [["dataset_synthetic", "dataset_iris"], ["qasm_simulator", "statevector_simulator"]]
    param_names = ["dataset", "backend name"]

    def setup_dataset_synthetic(self, X_train, X_test, y_train, num_inputs, quantum_instance_name):
        """Training VQC function for iris dataset."""

        feature_map = ZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)

        opflow_qnn = TwoLayerQNN(
            num_inputs,
            feature_map=feature_map,
            ansatz=ansatz,
            quantum_instance=self.backends[quantum_instance_name],
        )

        self.opflow_classifier_fitted = NeuralNetworkClassifier(opflow_qnn, optimizer=COBYLA())
        self.opflow_classifier_fitted.fit(X_train, y_train)

        self.y_predict = self.opflow_classifier_fitted.predict(X_test)

    def setup_dataset_iris(self, X_train, X_test, y_train, num_inputs, quantum_instance_name):
        """Training CircuitQNN function for iris dataset."""

        feature_map = ZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)

        opflow_qnn = TwoLayerQNN(
            num_inputs,
            feature_map=feature_map,
            ansatz=ansatz,
            quantum_instance=self.backends[quantum_instance_name],
        )

        self.opflow_classifier_fitted = NeuralNetworkClassifier(opflow_qnn, optimizer=COBYLA())
        self.opflow_classifier_fitted.fit(X_train, y_train)
        self.y_predict = self.opflow_classifier_fitted.predict(X_test)

    def setup(self, dataset, quantum_instance_name):
        """Setup"""

        self.X = self.datasets[dataset]["features"]
        num_inputs = len(self.X[0])
        self.y = self.datasets[dataset]["labels"]

        if dataset == "dataset_synthetic":
            self.y = 2 * self.y - 1  # in {-1, +1}
        elif dataset == "dataset_iris":
            # keeping only two classes as TwoLayerQNN only supports binary classification
            idx_binary_class = np.where(self.y != 2)[0]
            self.X = self.X[idx_binary_class]
            self.y = self.y[idx_binary_class]

            # scaling data
            scaler = MinMaxScaler((-1, 1))
            self.X = scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25
        )

        if dataset == "dataset_synthetic":
            self.setup_dataset_synthetic(
                self.X_train, self.X_test, self.y_train, num_inputs, quantum_instance_name
            )
        elif dataset == "dataset_iris":
            self.setup_dataset_iris(
                self.X_train, self.X_test, self.y_train, num_inputs, quantum_instance_name
            )

    def track_overall_accuracy_circuit_qnn_classifier(self, _, __):
        """Tracks the overall accuracy of the classification results."""
        acc_score = accuracy_score(self.y_test, self.y_predict)
        return acc_score

    def track_cohen_kappa_circuit_qnn_clasifier(self, _, __):
        """Tracks the cohen kappa score of the classification results."""
        cohen_kappa = cohen_kappa_score(self.y_test, self.y_predict)
        return cohen_kappa

    def track_f1_score_circuit_qnn_classifier(self, _, __):
        """Tracks the f1 score for each class of the classification results."""
        f1 = f1_score(self.y_test, self.y_predict, average="macro")
        return f1


if __name__ == "__main__":
    for dataset, backend in product(*OpflowQnnClassifierBenchmarks.params):
        bench = OpflowQnnClassifierBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue
