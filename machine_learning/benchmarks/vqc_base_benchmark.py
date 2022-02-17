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

"""Base class for VQC based classifier benchmarks."""
from abc import ABC
from typing import Optional

from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, VQC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from .base_classifier_benchmark import BaseClassifierBenchmark


class VqcBaseClassifierBenchmark(BaseClassifierBenchmark, ABC):
    """Base class for Opflow Classifier benchmarks."""

    def __init__(self) -> None:
        reshaper = FunctionTransformer(lambda x: x.reshape(-1, 1))
        encoder = OneHotEncoder(sparse=False)
        super().__init__(
            synthetic_label_encoder=Pipeline([("reshape", reshaper), ("one hot", encoder)]),
            iris_num_classes=2,
            iris_label_encoder=Pipeline([("reshape", reshaper), ("one hot", encoder)]),
        )

    def _construct_vqc_classifier_synthetic(
        self,
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None,
        loss_function: str = "cross_entropy",
    ) -> NeuralNetworkClassifier:
        """Training a VQC classifier for synthetic classification dataset."""
        return self._construct_vqc_classifier(2, quantum_instance_name, optimizer, loss_function)

    def _construct_vqc_classifier_iris(
        self,
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None,
        loss_function: str = "cross_entropy",
    ) -> NeuralNetworkClassifier:
        """Construct a VQC classifier for iris classification dataset."""
        return self._construct_vqc_classifier(4, quantum_instance_name, optimizer, loss_function)

    def _construct_vqc_classifier(
        self,
        num_inputs: int,
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None,
        loss_function: str = None,
    ) -> VQC:
        """Construct a VQC classifier."""
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)

        initial_point = algorithm_globals.random.random(ansatz.num_parameters)

        # construct variational quantum classifier
        model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            loss=loss_function,
            optimizer=optimizer,
            quantum_instance=self.backends[quantum_instance_name],
            initial_point=initial_point,
        )

        return model
