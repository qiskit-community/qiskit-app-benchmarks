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
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD, L_BFGS_B
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class OpflowQNNClassifierBenchmarks:
    """Opflow QNN Classifier benchmarks."""

    def __init__(self):
        quantum_instance_statevector = QuantumInstance(
            Aer.get_backend("statevector_simulator"), shots=1024
        )
        quantum_instance_qasm = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=1024)

        self.backends = {
            "statevector_simulator": quantum_instance_statevector,
            "qasm_simulator": quantum_instance_qasm,
        }

        self.optimizers = {"cobyla": COBYLA(), "nelder-mead": NELDER_MEAD(), "l-bfgs-b": L_BFGS_B()}

    timeout = 1200.0
    params = (["qasm_simulator", "statevector_simulator"], ["cobyla", "nelder-mead", "l-bfgs-b"])
    param_names = ["backend name", "optimizer"]

    def setup(self, quantum_instance_name, optimizer_name):
        """setup"""
        num_inputs = 2
        num_samples = 20

        seed = 50
        algorithm_globals.random_seed = seed
        np.random.default_rng(seed)

        self.X = 2 * np.random.rand(num_samples, num_inputs) - 1
        y01 = 1 * (np.sum(self.X, axis=1) >= 0)  # in { 0,  1}
        self.y = 2 * y01 - 1  # in {-1, +1}

        self.opflow_qnn = TwoLayerQNN(
            num_inputs, quantum_instance=self.backends[quantum_instance_name]
        )
        self.opflow_qnn.forward(self.X[0, :], np.random.rand(self.opflow_qnn.num_weights))

        self.opflow_classifier = NeuralNetworkClassifier(
            self.opflow_qnn, optimizer=self.optimizers[optimizer_name]
        )

        self.opflow_classifier_fitted = NeuralNetworkClassifier(
            self.opflow_qnn, optimizer=self.optimizers[optimizer_name]
        )
        self.opflow_classifier_fitted.fit(self.X, self.y)

    def time_fit_opflow_qnn_classifier(self, _, __):
        """Time fitting OpflowQNN classifier to data."""

        self.opflow_classifier.fit(self.X, self.y)

    def time_score_opflow_qnn_classifier(self, _, __):
        """Time scoring OpflowQNN classifier on data."""

        self.opflow_classifier_fitted.score(self.X, self.y)

    def time_predict_opflow_qnn_classifier(self, _, __):
        """Time predicting with classifier OpflowQNN."""

        y_predict = self.opflow_classifier_fitted.predict(self.X)
        return y_predict


if __name__ == "__main__":
    for backend, optimizer in product(*OpflowQNNClassifierBenchmarks.params):
        bench = OpflowQNNClassifierBenchmarks()
        try:
            bench.setup(backend, optimizer)
        except NotImplementedError:
            continue
        # we ensure the order: fit -> score -> predict
        for method in (
            "time_fit_opflow_qnn_classifier",
            "time_score_opflow_qnn_classifier",
            "time_predict_opflow_qnn_classifier",
        ):
            elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
