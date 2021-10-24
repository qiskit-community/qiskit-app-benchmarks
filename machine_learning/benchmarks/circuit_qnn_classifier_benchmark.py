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

"""Circuit QNN Classifier benchmarks."""
from itertools import product
from timeit import timeit

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class CircuitQnnClassifierBenchmarks(BaseClassifierBenchmark):
    """Circuit QNN Classifier benchmarks."""

    version = 1
    timeout = 1200.0
    params = [["dataset_synthetic"], ["qasm_simulator", "statevector_simulator"]]
    param_names = ["backend name"]

    def __init__(self):
        super().__init__()

        self.output_shape = 2  # corresponds to the number of classes, possible outcomes of the (
        # parity) mapping.

    def setup(self, dataset, quantum_instance_name):
        """setup"""
        self.X = self.datasets[dataset]
        num_inputs = len(self.X[0])
        self.y01 = 1 * (np.sum(self.X, axis=1) >= 0)  # in { 0,  1}

        # parity maps bitstrings to 0 or 1
        def parity(x):
            return f"{x:b}".count("1") % 2

        # construct feature map
        feature_map = ZZFeatureMap(num_inputs)

        # construct ansatz
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct quantum circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        # construct QNN
        self.circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=self.output_shape,
            quantum_instance=self.backends[quantum_instance_name],
        )

        self.circuit_classifier_fitted = NeuralNetworkClassifier(
            neural_network=self.circuit_qnn, optimizer=COBYLA()
        )
        self.circuit_classifier_fitted.fit(self.X, self.y01)

    def time_score_circuit_qnn_classifier(self, _, __):
        """Time scoring CircuitQNN classifier on data."""

        self.circuit_classifier_fitted.score(self.X, self.y01)

    def time_predict_circuit_qnn_classifier(self, _, __):
        """Time predicting with CircuitQNN classifier."""

        y_predict = self.circuit_classifier_fitted.predict(self.X)
        return y_predict


if __name__ == "__main__":
    for dataset, backend in product(*CircuitQnnClassifierBenchmarks.params):
        bench = CircuitQnnClassifierBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue

        for method in (
            "time_score_circuit_qnn_classifier",
            "time_predict_circuit_qnn_classifier",
        ):
            elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
