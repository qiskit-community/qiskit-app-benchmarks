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

"""Neural Network Regressor benchmarks."""
from itertools import product
from timeit import timeit

from qiskit.circuit.library import EfficientSU2, ZFeatureMap
from qiskit.algorithms.optimizers import L_BFGS_B, NELDER_MEAD, COBYLA
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from sklearn.preprocessing import MinMaxScaler

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_regressor_benchmark import BaseRegressorBenchmark


class OpflowQnnFitRegressorBenchmarks(BaseRegressorBenchmark):
    """Opflow QNN Regressor benchmarks."""

    version = 1
    timeout = 1200.0
    params = (
        ["dataset_synthetic_regression", "dataset_ccpp"],
        ["qasm_simulator", "statevector_simulator"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
    )
    param_names = ["dataset", "backend name", "optimizer"]

    def __init__(self):
        super().__init__()
        self.optimizers = {"cobyla": COBYLA(), "nelder-mead": NELDER_MEAD(), "l-bfgs-b": L_BFGS_B()}

    def setup_dataset_synthetic_regression(self, quantum_instance_name, optimizer):
        """Training Opflow QNN function for synthetic regression dataset."""

        num_inputs = 2
        feature_map = ZFeatureMap(num_inputs)
        ansatz = EfficientSU2(num_inputs)

        opflow_qnn = TwoLayerQNN(
            num_inputs, feature_map, ansatz, quantum_instance=self.backends[quantum_instance_name]
        )

        self.opflow_regressor_fitted = NeuralNetworkRegressor(
            opflow_qnn, optimizer=self.optimizers[optimizer]
        )

    def setup_dataset_ccpp(self, X, y, quantum_instance_name, optimizer):
        """Training Opflow QNN for CCPP dataset."""

        scaler = MinMaxScaler((-1, 1))
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y.reshape(-1, 1))

        num_inputs = 4
        feature_map = ZFeatureMap(num_inputs)
        ansatz = EfficientSU2(num_inputs)

        opflow_qnn = TwoLayerQNN(
            num_inputs, feature_map, ansatz, quantum_instance=self.backends[quantum_instance_name]
        )

        self.opflow_regressor_fitted = NeuralNetworkRegressor(
            opflow_qnn, optimizer=self.optimizers[optimizer]
        )

    def setup(self, dataset, quantum_instance_name, optimizer):
        """setup"""

        self.X = self.datasets[dataset]["features"]
        self.y = self.datasets[dataset]["labels"]

        if dataset == "dataset_synthetic_regression":
            self.setup_dataset_synthetic_regression(quantum_instance_name, optimizer)
        elif dataset == "dataset_ccpp":
            self.setup_dataset_ccpp(self.X, self.y, quantum_instance_name, optimizer)

    def time_fit_opflow_qnn_regressor(self, _, __, ___):
        """Time fitting OpflowQNN regressor to data."""

        self.opflow_regressor_fitted.fit(self.X, self.y)


if __name__ == "__main__":
    for dataset, backend, optimizer in product(*OpflowQnnFitRegressorBenchmarks.params):
        bench = OpflowQnnFitRegressorBenchmarks()
        try:
            bench.setup(dataset, backend, optimizer)
        except NotImplementedError:
            continue

        for method in ["time_fit_opflow_qnn_regressor"]:
            elapsed = timeit(f"bench.{method}(None, None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
