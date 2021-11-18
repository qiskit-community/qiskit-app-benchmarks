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

"""Variational Quantum Classifier benchmarks."""
from itertools import product
from timeit import timeit

import numpy as np
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD
from qiskit_machine_learning.algorithms import VQC

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class VqcFitBenchmarks(BaseClassifierBenchmark):
    """Variational Quantum Classifier benchmarks."""

    version = 1
    timeout = 1200.0
    params = (
        ["dataset_synthetic", "dataset_iris"],
        ["qasm_simulator", "statevector_simulator"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
        ["cross_entropy", "squared_error"],
    )
    param_names = ["dataset", "backend name", "optimizer", "loss function"]

    def __init__(self):
        super().__init__()

        self.optimizers = {"cobyla": COBYLA(), "nelder-mead": NELDER_MEAD(), "l-bfgs-b": L_BFGS_B()}

    def setup_dataset_synthetic_classification(
        self, y, quantum_instance_name, optimizer_name, loss_name
    ):
        """Training VQC for synthetic classification dataset."""

        num_inputs = 2

        self.y_one_hot = np.zeros((len(y), 2))
        for i, _ in enumerate(y):
            self.y_one_hot[i, y[i]] = 1

        # construct feature map, ansatz, and optimizer
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct variational quantum classifier
        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            loss=loss_name,
            optimizer=self.optimizers[optimizer_name],
            quantum_instance=self.backends[quantum_instance_name],
        )

    def setup_dataset_iris(self, y, quantum_instance_name, optimizer_name, loss_name):
        """Training VQC for iris dataset."""

        num_inputs = 4

        self.y_one_hot = np.zeros((len(y), 3))
        for i, _ in enumerate(y):
            self.y_one_hot[i, y[i]] = 1

        # construct feature map, ansatz, and optimizer
        feature_map = ZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            loss=loss_name,
            optimizer=self.optimizers[optimizer_name],
            quantum_instance=self.backends[quantum_instance_name],
        )

    def setup(self, dataset, quantum_instance_name, optimizer_name, loss_name):
        """setup"""

        self.X = self.datasets[dataset]["features"]
        self.y = self.datasets[dataset]["labels"]

        if dataset == "dataset_synthetic":
            self.setup_dataset_synthetic_classification(
                self.y, quantum_instance_name, optimizer_name, loss_name
            )
        elif dataset == "dataset_iris":
            self.setup_dataset_iris(self.y, quantum_instance_name, optimizer_name, loss_name)

    def time_fit_vqc(self, _, __, ___, ____):
        """Time fitting VQC to data."""

        self.vqc.fit(self.X, self.y_one_hot)


if __name__ == "__main__":
    for dataset, backend, optimizer, loss_function in product(*VqcFitBenchmarks.params):
        bench = VqcFitBenchmarks()
        try:
            bench.setup(dataset, backend, optimizer, loss_function)
        except NotImplementedError:
            continue

        for method in ["time_fit_vqc"]:
            elapsed = timeit(
                f"bench.{method}(None, None, None, None)", number=10, globals=globals()
            )
            print(f"{method}:\t{elapsed}")
