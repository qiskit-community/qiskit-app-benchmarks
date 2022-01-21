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

"""Base class for Opflow based classifier benchmarks."""
from abc import ABC
from typing import Optional

from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from sklearn.preprocessing import FunctionTransformer

from .base_classifier_benchmark import BaseClassifierBenchmark


class OpflowQnnBaseClassifierBenchmark(BaseClassifierBenchmark, ABC):
    """Base class for Opflow Classifier benchmarks."""

    def __init__(self) -> None:
        encoder = FunctionTransformer(lambda x: 2 * x - 1)
        super().__init__(
            synthetic_label_encoder=encoder, iris_num_classes=2, iris_label_encoder=encoder
        )

    def _construct_opflow_classifier_synthetic(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkClassifier:
        """Training a TwoLayerQNN-based classifier for synthetic classification dataset."""
        return self._construct_opflow_classifier(2, quantum_instance_name, optimizer)

    def _construct_opflow_classifier_iris(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkClassifier:
        """Construct a TwoLayerQNN-based classifier for iris classification dataset."""
        return self._construct_opflow_classifier(4, quantum_instance_name, optimizer)

    def _construct_opflow_classifier(
        self, num_inputs: int, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkClassifier:
        """Construct a TwoLayerQNN-based classifier."""
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)

        opflow_qnn = TwoLayerQNN(
            num_inputs,
            feature_map=feature_map,
            ansatz=ansatz,
            quantum_instance=self.backends[quantum_instance_name],
        )

        initial_point = algorithm_globals.random.random(ansatz.num_parameters)
        model = NeuralNetworkClassifier(
            opflow_qnn, optimizer=optimizer, initial_point=initial_point
        )
        return model
