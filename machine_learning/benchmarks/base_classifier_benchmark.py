# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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
from typing import Tuple, Optional, Union

import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from sklearn.base import TransformerMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from .datasets import (
    DATASET_SYNTHETIC_CLASSIFICATION_FEATURES,
    DATASET_SYNTHETIC_CLASSIFICATION_LABELS,
)

DATASET_SYNTHETIC_CLASSIFICATION = "dataset_synthetic"
DATASET_IRIS_CLASSIFICATION = "dataset_iris"


class BaseClassifierBenchmark(ABC):
    """Base for Classifier benchmarks."""

    def __init__(
        self,
        synthetic_label_encoder: Optional[Union[TransformerMixin, Pipeline]] = None,
        iris_num_classes: int = 3,
        iris_label_encoder: Optional[Union[TransformerMixin, Pipeline]] = None,
    ):
        algorithm_globals.random_seed = 12345
        quantum_instance_statevector = QuantumInstance(
            Aer.get_backend("statevector_simulator"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        quantum_instance_qasm = QuantumInstance(
            Aer.get_backend("qasm_simulator"),
            shots=1024,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        self.backends = {
            "statevector_simulator": quantum_instance_statevector,
            "qasm_simulator": quantum_instance_qasm,
        }

        # if none, just identity transformer
        synthetic_label_encoder = synthetic_label_encoder or FunctionTransformer()
        (
            synth_train_features,
            synth_test_features,
            synth_train_labels,
            synth_test_labels,
        ) = self._prepare_synthetic(synthetic_label_encoder)

        iris_label_encoder = iris_label_encoder or FunctionTransformer()
        (
            iris_train_features,
            iris_test_features,
            iris_train_labels,
            iris_test_labels,
        ) = self._prepare_iris(iris_num_classes, iris_label_encoder)

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

    def _prepare_synthetic(
        self, label_encoder: Union[TransformerMixin, Pipeline]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        synthetic_labels = label_encoder.fit_transform(DATASET_SYNTHETIC_CLASSIFICATION_LABELS)

        (
            synth_train_features,
            synth_test_features,
            synth_train_labels,
            synth_test_labels,
        ) = train_test_split(
            DATASET_SYNTHETIC_CLASSIFICATION_FEATURES,
            synthetic_labels,
            test_size=5,
            shuffle=False,
        )
        return synth_train_features, synth_test_features, synth_train_labels, synth_test_labels

    def _prepare_iris(
        self, num_classes: int, label_encoder: Union[TransformerMixin, Pipeline]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        iris_features_all, iris_labels_all = load_iris(return_X_y=True)
        size = 25
        iris_features = np.zeros((size, 4))
        iris_labels = np.zeros(size)
        for i in range(size):
            # there are 50 samples of each class, three classes, but we sample only two!
            index = 50 * (i % num_classes) + i
            iris_features[i, :] = iris_features_all[index]
            iris_labels[i] = iris_labels_all[index]
        scaler = MinMaxScaler((-1, 1))
        iris_features = scaler.fit_transform(iris_features)
        iris_labels = label_encoder.fit_transform(iris_labels)

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

        return iris_train_features, iris_test_features, iris_train_labels, iris_test_labels
