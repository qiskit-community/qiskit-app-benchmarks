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
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class VqcBenchmarks(BaseClassifierBenchmark):
    """Variational Quantum Classifier benchmarks."""

    version = 1
    timeout = 1200.0
    params = [["dataset_1"], ["qasm_simulator", "statevector_simulator"]]
    param_names = ["backend name"]

    def setup(self, dataset, quantum_instance_name):
        """setup"""
        self.X = self.datasets[dataset]["features"]
        num_inputs = len(self.X[0])
        num_samples = len(self.X)
        y01 = self.datasets[dataset]["labels"]
        self.y_one_hot = np.zeros((num_samples, 2))
        for i in range(num_samples):
            self.y_one_hot[i, y01[i]] = 1

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

        self.vqc_fitted.fit(self.X, self.y_one_hot)

    def time_score_vqc(self, _, __):
        """Time scoring VQC on data."""

        self.vqc_fitted.score(self.X, self.y_one_hot)

    def time_predict_vqc(self, _, __):
        """Time predicting with VQC."""

        y_predict = self.vqc_fitted.predict(self.X)
        return y_predict


if __name__ == "__main__":
    for dataset, backend in product(*VqcBenchmarks.params):
        bench = VqcBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue

        for method in ("time_score_vqc", "time_predict_vqc"):
            elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
