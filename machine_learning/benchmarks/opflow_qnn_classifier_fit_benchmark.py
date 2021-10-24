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
from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD, L_BFGS_B
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class OpflowQnnFitClassifierBenchmarks(BaseClassifierBenchmark):
    """Opflow QNN Classifier benchmarks."""

    version = 1
    timeout = 1200.0
    params = (
        ["dataset_synthetic"],
        ["qasm_simulator", "statevector_simulator"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
    )
    param_names = ["backend name", "optimizer"]

    def __init__(self):
        super().__init__()
        self.optimizers = {"cobyla": COBYLA(), "nelder-mead": NELDER_MEAD(), "l-bfgs-b": L_BFGS_B()}

    def setup(self, dataset, quantum_instance_name, optimizer_name):
        """setup"""
        self.X = self.datasets[dataset]
        num_inputs = len(self.X[0])
        y01 = 1 * (np.sum(self.X, axis=1) >= 0)  # in { 0,  1}
        self.y = 2 * y01 - 1  # in {-1, +1}

        opflow_qnn = TwoLayerQNN(num_inputs, quantum_instance=self.backends[quantum_instance_name])
        opflow_qnn.forward(self.X[0, :], np.random.rand(opflow_qnn.num_weights))

        self.opflow_classifier = NeuralNetworkClassifier(
            opflow_qnn, optimizer=self.optimizers[optimizer_name]
        )

    def time_fit_opflow_qnn_classifier(self, _, __, ___):
        """Time fitting OpflowQNN classifier to data."""

        self.opflow_classifier.fit(self.X, self.y)


if __name__ == "__main__":
    for dataset, backend, optimizer in product(*OpflowQnnFitClassifierBenchmarks.params):
        bench = OpflowQnnFitClassifierBenchmarks()
        try:
            bench.setup(dataset, backend, optimizer)
        except NotImplementedError:
            continue

        for method in ["time_fit_opflow_qnn_classifier"]:
            elapsed = timeit(f"bench.{method}(None, None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
