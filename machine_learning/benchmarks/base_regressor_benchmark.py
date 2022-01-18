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
"""Base class for Regressor benchmarks."""

from abc import ABC
from typing import Optional

from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .datasets import (
    DATASET_SYNTHETIC_REGRESSION_FEATURES,
    DATASET_SYNTHETIC_REGRESSION_LABELS,
    load_ccpp,
)

DATASET_SYNTHETIC_REGRESSION = "dataset_synthetic_regression"
DATASET_CCPP_REGRESSION = "dataset_ccpp"


class BaseRegressorBenchmark(ABC):
    """Base class for Regressor benchmarks."""

    def __init__(self):
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

        # prepare synthetic dataset
        (
            synth_train_features,
            synth_test_features,
            synth_train_labels,
            synth_test_labels,
        ) = train_test_split(
            DATASET_SYNTHETIC_REGRESSION_FEATURES,
            DATASET_SYNTHETIC_REGRESSION_LABELS,
            test_size=5,
            shuffle=False,
        )

        # prepare CCPP dataset, we can afford only a tiny subset of the dataset for training
        ccpp_features, ccpp_labels = load_ccpp()
        ccpp_features = ccpp_features[:25]
        ccpp_labels = ccpp_labels[:25]

        scaler = MinMaxScaler((-1, 1))
        ccpp_features = scaler.fit_transform(ccpp_features)
        ccpp_labels = scaler.fit_transform(ccpp_labels.reshape(-1, 1))

        (
            ccpp_train_features,
            ccpp_test_features,
            ccpp_train_labels,
            ccpp_test_labels,
        ) = train_test_split(ccpp_features, ccpp_labels, test_size=5, shuffle=False)

        self.datasets = {
            DATASET_SYNTHETIC_REGRESSION: {
                "train_features": synth_train_features,
                "train_labels": synth_train_labels,
                "test_features": synth_test_features,
                "test_labels": synth_test_labels,
            },
            DATASET_CCPP_REGRESSION: {
                "train_features": ccpp_train_features,
                "train_labels": ccpp_train_labels,
                "test_features": ccpp_test_features,
                "test_labels": ccpp_test_labels,
            },
        }

    def _construct_qnn_synthetic(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkRegressor:
        num_inputs = 1
        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        # construct simple ansatz
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)

        opflow_qnn = TwoLayerQNN(
            num_inputs, feature_map, ansatz, quantum_instance=self.backends[quantum_instance_name]
        )

        return NeuralNetworkRegressor(opflow_qnn, optimizer=optimizer)

    def _construct_qnn_ccpp(
        self, quantum_instance_name: str, optimizer: Optional[Optimizer] = None
    ) -> NeuralNetworkRegressor:
        num_inputs = 4
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs)
        opflow_qnn = TwoLayerQNN(
            num_inputs, feature_map, ansatz, quantum_instance=self.backends[quantum_instance_name]
        )

        model = NeuralNetworkRegressor(opflow_qnn, optimizer=optimizer)
        return model
