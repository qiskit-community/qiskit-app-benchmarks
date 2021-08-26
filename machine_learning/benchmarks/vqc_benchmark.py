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
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class VQCBenchmarks:
    """Variational Quantum Classifier benchmarks."""

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

        # construct feature map, ansatz, and optimizer
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct variational quantum classifier
        self.vqc = VQC(feature_map=feature_map,
                       ansatz=ansatz,
                       loss='cross_entropy',
                       optimizer=COBYLA(),
                       quantum_instance=self.backends[quantum_instance_name])

        # fit classifier to data
        self.vqc_fitted = VQC(feature_map=feature_map,
                              ansatz=ansatz,
                              loss='cross_entropy',
                              optimizer=COBYLA(),
                              quantum_instance=self.backends[quantum_instance_name])
        self.vqc_fitted.fit(self.X, self.y01)

        # score classifier
        self.vqc_scored = VQC(feature_map=feature_map,
                              ansatz=ansatz,
                              loss='cross_entropy',
                              optimizer=COBYLA(),
                              quantum_instance=self.backends[quantum_instance_name])
        self.vqc_scored.fit(self.X, self.y01)
        self.vqc_scored.score(self.X, self.y01)

    def time_fit_vqc(self, _):
        """Time fitting VQC to data."""

        self.vqc.fit(self.X, self.y)

    def time_score_vqc(self, _):
        """Time scoring VQC on data."""

        self.vqc_fitted.score(self.X, self.y)

    def time_predict_vqc(self, _):
        """Time predicting with VQC."""

        y_predict = self.vqc_scored.predict(self.X)
        return y_predict


if __name__ == "__main__":
    for backend in VQCBenchmarks.params:
        bench = VQCBenchmarks()
        try:
            bench.setup(backend)
        except NotImplementedError:
            continue
        for method in set(dir(VQCBenchmarks)):
            if method.startswith("time_"):
                elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
                print("f{method}:\t{elapsed}")
