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

"""Base for Classifier benchmarks."""

from abc import ABC
from typing import Optional

from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import CircuitQNN

from .datasets import (
    DATASET_SYNTHETIC_CLASSIFICATION_FEATURES,
    DATASET_SYNTHETIC_CLASSIFICATION_LABELS,
    DATASET_IRIS_CLASSIFICATION_FEATURES,
    DATASET_IRIS_CLASSIFICATION_LABELS,
)


class BaseClassifierBenchmark(ABC):
    """Base for Classifier benchmarks."""

    def __init__(self):

        quantum_instance_statevector = QuantumInstance(
            Aer.get_backend("statevector_simulator"), shots=1024
        )
        quantum_instance_qasm = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=1024)

        self.backends = {
            "statevector_simulator": quantum_instance_statevector,
            "qasm_simulator": quantum_instance_qasm,
        }

        self.datasets = {
            "dataset_synthetic": {
                "features": DATASET_SYNTHETIC_CLASSIFICATION_FEATURES,
                "labels": DATASET_SYNTHETIC_CLASSIFICATION_LABELS,
            },
            "dataset_iris": {
                "features": DATASET_IRIS_CLASSIFICATION_FEATURES,
                "labels": DATASET_IRIS_CLASSIFICATION_LABELS,
            },
        }

    def _construct_qnn_synthetic(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkClassifier:

        output_shape = 2
        num_inputs = 2

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
            output_shape=output_shape,
            quantum_instance=self.backends[quantum_instance_name],
        )

        model = NeuralNetworkClassifier(neural_network=self.circuit_qnn, optimizer=optimizer)
        return model

    def _construct_qnn_iris(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkClassifier:
        output_shape = 3
        num_inputs = 4

        # creating feature map
        feature_map = ZZFeatureMap(num_inputs)

        # creating ansatz
        ansatz = RealAmplitudes(num_inputs)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        def three_class(x):
            return f"{x:b}".count("1") % 3

        # construct QNN
        circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=three_class,
            output_shape=output_shape,
            quantum_instance=self.backends[quantum_instance_name],
        )

        model = NeuralNetworkClassifier(neural_network=circuit_qnn,optimizer=optimizer)
        return model
