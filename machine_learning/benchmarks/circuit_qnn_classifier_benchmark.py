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
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class CircuitQNNClassifierBenchmarks:
    """Circuit QNN Classifier benchmarks."""

    def __init__(self):
        quantum_instance_statevector = QuantumInstance(Aer.get_backend('statevector_simulator'),
                                                       shots=1024)
        quantum_instance_qasm = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

        self.backends = {'statevector_simulator': quantum_instance_statevector,
                         'qasm_simulator': quantum_instance_qasm}

    timeout = 1200.0
    params = ['qasm_simulator', 'statevector_simulator']
    param_names = ["backend name"]

    def setup(self, quantum_instance_name):
        """setup"""
        num_inputs = 2
        num_samples = 20

        seed = 50
        algorithm_globals.random_seed = seed

        self.X = 2 * np.random.rand(num_samples, num_inputs) - 1
        self.y01 = 1 * (np.sum(self.X, axis=1) >= 0)  # in { 0,  1}
        self.y = 2 * self.y01 - 1  # in {-1, +1}

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
            return '{:b}'.format(x).count('1') % 2

        output_shape = 2  # corresponds to the number of classes, possible outcomes of the (
        # parity) mapping.

        # construct QNN
        circuit_qnn = CircuitQNN(circuit=qc,
                                 input_params=feature_map.parameters,
                                 weight_params=ansatz.parameters,
                                 interpret=parity,
                                 output_shape=output_shape,
                                 quantum_instance=self.backends[quantum_instance_name])

        # construct classifier
        self.circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn,
                                                          optimizer=COBYLA())

        # fit classifier to data
        self.circuit_classifier_fitted = NeuralNetworkClassifier(neural_network=circuit_qnn,
                                                                 optimizer=COBYLA())
        self.circuit_classifier_fitted.fit(self.X, self.y01)

        # score classifier
        self.circuit_classifier_scored = NeuralNetworkClassifier(neural_network=circuit_qnn,
                                                                 optimizer=COBYLA())
        self.circuit_classifier_scored.fit(self.X, self.y01)
        self.circuit_classifier_scored.score(self.X, self.y01)

    def time_fit_circuit_qnn(self, _):
        """Time fitting OpflowQNN to data."""

        self.circuit_classifier.fit(self.X, self.y01)

    def time_score_circuit_qnn(self, _):
        """Time scoring OpflowQNN on data."""

        self.circuit_classifier_fitted.score(self.X, self.y01)

    def time_predict_circuit_qnn(self, _):
        """Time predicting with OpflowQNN."""

        y_predict = self.circuit_classifier_scored.predict(self.X)
        return y_predict


if __name__ == "__main__":
    for backend in CircuitQNNClassifierBenchmarks.params:
        bench = CircuitQNNClassifierBenchmarks()
        try:
            bench.setup(backend)
        except NotImplementedError:
            continue
        for method in set(dir(CircuitQNNClassifierBenchmarks)):
            if method.startswith("time_"):
                elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
                print("f{method}:\t{elapsed}")
