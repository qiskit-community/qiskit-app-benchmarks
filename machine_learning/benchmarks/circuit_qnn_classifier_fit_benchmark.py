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
from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD, L_BFGS_B
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class CircuitQnnFitClassifierBenchmarks(BaseClassifierBenchmark):
    """Circuit QNN Classifier benchmarks."""

    def __init__(self):
        super().__init__()

        self.output_shape = 2  # corresponds to the number of classes, possible outcomes of the (

        self.optimizers = {"cobyla": COBYLA(), "nelder-mead": NELDER_MEAD(), "l-bfgs-b": L_BFGS_B()}

    timeout = 1200.0
    params = (
        ["dataset_1"],
        ["qasm_simulator", "statevector_simulator"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
    )
    param_names = ["backend name", "optimizer"]

    def setup(self, dataset, quantum_instance_name, optimizer_name):
        """setup"""
        self.X = self.datasets[dataset]
        num_inputs = len(self.X[0])
        self.y01 = 1 * (np.sum(self.X, axis=1) >= 0)  # in { 0,  1}

        # construct feature map
        feature_map = ZZFeatureMap(num_inputs)

        # construct ansatz
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct quantum circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        # parity maps bitstrings to 0 or 1
        def parity(x):
            return "{:b}".format(x).count("1") % 2

        # construct QNN
        self.circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=self.output_shape,
            quantum_instance=self.backends[quantum_instance_name],
        )

        # construct classifier
        self.circuit_classifier = NeuralNetworkClassifier(
            neural_network=self.circuit_qnn, optimizer=self.optimizers[optimizer_name]
        )

    def time_fit_circuit_qnn_classifier(self, _, __, ___):
        """Time fitting CircuitQNN classifier to data."""

        self.circuit_classifier.fit(self.X, self.y01)


if __name__ == "__main__":
    for dataset, backend, optimizer in product(*CircuitQnnFitClassifierBenchmarks.params):
        bench = CircuitQnnFitClassifierBenchmarks()
        try:
            bench.setup(dataset, backend, optimizer)
        except NotImplementedError:
            continue

        for method in ["time_fit_circuit_qnn_classifier"]:
            elapsed = timeit(f"bench.{method}(None, None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
