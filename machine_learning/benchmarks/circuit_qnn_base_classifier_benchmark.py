# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base class for CircuitQNN based Classifier benchmarks."""
from abc import ABC
from typing import Optional, Callable

from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import CircuitQNN

from .base_classifier_benchmark import BaseClassifierBenchmark


class CircuitQnnBaseClassifierBenchmark(BaseClassifierBenchmark, ABC):
    """Base class for CircuitQNN Classifier benchmarks."""

    def __init__(self) -> None:
        super().__init__()

    def _construct_qnn_classifier_synthetic(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkClassifier:

        # parity maps bitstrings to 0 or 1
        def parity(x):
            return f"{x:b}".count("1") % 2

        return self._construct_qnn_classifier(
            num_inputs=2,
            output_shape=2,
            interpret=parity,
            quantum_instance_name=quantum_instance_name,
            optimizer=optimizer,
        )

    def _construct_qnn_classifier_iris(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkClassifier:

        # map to three classes
        def three_class(x):
            return f"{x:b}".count("1") % 3

        return self._construct_qnn_classifier(
            num_inputs=4,
            output_shape=3,
            interpret=three_class,
            quantum_instance_name=quantum_instance_name,
            optimizer=optimizer,
        )

    def _construct_qnn_classifier(
        self,
        num_inputs: int,
        output_shape: int,
        interpret: Callable[[int], int],
        quantum_instance_name: str,
        optimizer: Optional[Optimizer],
    ) -> NeuralNetworkClassifier:
        feature_map = ZZFeatureMap(num_inputs)

        ansatz = RealAmplitudes(num_inputs)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=interpret,
            output_shape=output_shape,
            quantum_instance=self.backends[quantum_instance_name],
        )
        initial_point = algorithm_globals.random.random(ansatz.num_parameters)
        model = NeuralNetworkClassifier(
            neural_network=circuit_qnn, optimizer=optimizer, initial_point=initial_point
        )
        return model
