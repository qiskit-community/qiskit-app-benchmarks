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

"""Base for VQC based classifier benchmarks."""
from typing import Optional

import numpy as np
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, VQC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .base_classifier_benchmark import BaseClassifierBenchmark
from .datasets import (
    DATASET_SYNTHETIC_CLASSIFICATION_FEATURES,
    DATASET_SYNTHETIC_CLASSIFICATION_LABELS,
)


class VqcBaseClassifierBenchmark(BaseClassifierBenchmark):
    """Base for Opflow Classifier benchmarks."""

    def __init__(self):
        super().__init__()

        # prepare synthetic
        (
            synth_train_features,
            synth_test_features,
            synth_train_labels,
            synth_test_labels,
        ) = train_test_split(
            DATASET_SYNTHETIC_CLASSIFICATION_FEATURES,
            DATASET_SYNTHETIC_CLASSIFICATION_LABELS,
            test_size=5,
            shuffle=False,
        )

        # one hot encoding for VQC
        encoder = OneHotEncoder()
        # VQC does not work with csr_matrix returned by the encoder
        synth_train_labels = encoder.fit_transform(synth_train_labels.reshape(-1, 1)).toarray()
        synth_test_labels = encoder.fit_transform(synth_test_labels.reshape(-1, 1)).toarray()

        # prepare iris
        iris_features_all, iris_labels_all = load_iris(return_X_y=True)

        size = 25
        iris_features = np.zeros((size, 4))
        iris_labels = np.zeros(size)

        for i in range(25):
            # there are 50 samples of each class, three classes, but we sample only two!
            index = 50 * (i % 3) + i
            iris_features[i, :] = iris_features_all[index]
            iris_labels[i] = iris_labels_all[index]

        scaler = MinMaxScaler((-1, 1))
        iris_features = scaler.fit_transform(iris_features)
        # one hot encoding
        iris_labels = encoder.fit_transform(iris_labels.reshape(-1, 1)).toarray()

        (
            iris_train_features,
            iris_test_features,
            iris_train_labels,
            iris_test_labels,
        ) = train_test_split(
            iris_features,
            iris_labels,
            test_size=5,
            shuffle=False,
        )

        self.datasets = {
            "dataset_synthetic": {
                "train_features": synth_train_features,
                "train_labels": synth_train_labels,
                "test_features": synth_test_features,
                "test_labels": synth_test_labels,
            },
            "dataset_iris": {
                "train_features": iris_train_features,
                "train_labels": iris_train_labels,
                "test_features": iris_test_features,
                "test_labels": iris_test_labels,
            },
        }

    def _construct_vqc_classifier_synthetic(
        self,
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None,
        loss_function: str = None,
    ) -> NeuralNetworkClassifier:
        """Training a VQC classifier for synthetic classification dataset."""
        return self._construct_vqc_classifier(2, quantum_instance_name, optimizer, loss_function)

    def _construct_vqc_classifier_iris(
        self,
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None,
        loss_function: str = None,
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
