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

"""VQC score benchmarks."""
from itertools import product

import numpy as np
from joblib import dump, load
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class VqcScoreClassifierBenchmarks(BaseClassifierBenchmark):
    """VQC Score benchmarks."""

    version = 1
    timeout = 1200.0
    params = [["dataset_synthetic", "dataset_iris"], ["qasm_simulator", "statevector_simulator"]]
    param_names = ["dataset", "backend name"]

    def setup_dataset_synthetic(self, X_train, X_test, y_train, num_inputs, quantum_instance_name):
        """Training VQC function for iris dataset."""

        num_inputs = 2

        # construct feature map, ansatz, and optimizer
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct variational quantum classifier
        self.vqc_fitted = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            loss="cross_entropy",
            optimizer=COBYLA(),
            quantum_instance=self.backends[quantum_instance_name],
        )

        try:
            self.vqc_fitted._fit_result = load(
                f"/tmp/dataset_synthetic_classification_{quantum_instance_name}.obj"
            )
        except FileNotFoundError:
            self.vqc_fitted.fit(X_train, y_train)

        self.y_predict = self.vqc_fitted.predict(X_test)

    def setup_dataset_iris(self, X_train, X_test, y_train, num_inputs, quantum_instance_name):
        """Training CircuitQNN function for iris dataset."""

        num_inputs = 4

        # construct feature map, ansatz, and optimizer
        feature_map = ZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct variational quantum classifier
        self.vqc_fitted = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            loss="cross_entropy",
            optimizer=L_BFGS_B(),
            quantum_instance=self.backends[quantum_instance_name],
        )

        try:
            self.vqc_fitted._fit_result = load(f"/tmp/dataset_iris_{quantum_instance_name}.obj")
        except FileNotFoundError:
            self.vqc_fitted.fit(X_train, y_train)

        self.y_predict = self.vqc_fitted.predict(X_test)

    def setup(self, dataset, quantum_instance_name):
        """Setup"""

        self.X = self.datasets[dataset]["features"]
        num_inputs = len(self.X[0])
        self.y = self.datasets[dataset]["labels"]

        if dataset == "dataset_synthetic":
            # one hot encoding target
            self.y_one_hot = np.zeros((len(self.y), 2))
            for i, _ in enumerate(self.y):
                self.y_one_hot[i, self.y[i]] = 1

        elif dataset == "dataset_iris":
            # scaling data
            scaler = MinMaxScaler((-1, 1))
            self.X = scaler.fit_transform(self.X)

            # one hot encoding target
            self.y_one_hot = np.zeros((len(self.y), 3))
            for i, _ in enumerate(self.y):
                self.y_one_hot[i, self.y[i]] = 1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_one_hot, test_size=0.25
        )

        if dataset == "dataset_synthetic":
            self.setup_dataset_synthetic(
                self.X_train, self.X_test, self.y_train, num_inputs, quantum_instance_name
            )
        elif dataset == "dataset_iris":
            self.setup_dataset_iris(
                self.X_train, self.X_test, self.y_train, num_inputs, quantum_instance_name
            )

    def setup_cache(self):
        """Cache VQC fitted model"""
        for dataset, backend in product(*self.params):
            self.setup(dataset, backend)

            dump(self.vqc_fitted._fit_result, f"/tmp/{dataset}_{backend}.obj")

    def track_overall_accuracy_circuit_qnn_classifier(self, _, __):
        """Tracks the overall accuracy of the classification results."""
        acc_score = accuracy_score(self.y_test.argmax(axis=1), self.y_predict.argmax(axis=1))
        return acc_score

    def track_cohen_kappa_circuit_qnn_clasifier(self, _, __):
        """Tracks the cohen kappa score of the classification results."""
        cohen_kappa = cohen_kappa_score(self.y_test.argmax(axis=1), self.y_predict.argmax(axis=1))
        return cohen_kappa

    def track_f1_score_circuit_qnn_classifier(self, _, __):
        """Tracks the f1 score for each class of the classification results."""
        f1 = f1_score(self.y_test.argmax(axis=1), self.y_predict.argmax(axis=1), average="macro")
        return f1


if __name__ == "__main__":
    for dataset, backend in product(*VqcScoreClassifierBenchmarks.params):
        bench = VqcScoreClassifierBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue
