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

"""Variational Quantum Regressor benchmarks."""
from itertools import product
from timeit import timeit

from sklearn.preprocessing import MinMaxScaler
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD
from qiskit.circuit.library import ZFeatureMap, PauliTwoDesign
from qiskit_machine_learning.algorithms import VQR

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_regressor_benchmark import BaseRegressorBenchmark


class VqrFitBenchmarks(BaseRegressorBenchmark):
    """Variational Quantum Regressor benchmarks."""

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

    def setup_dataset_synthetic_regression(self, quantum_instance_name, optimizer_name):
        """Training VQR function for synthetic regression dataset."""

        feature_map = ZFeatureMap(2)
        ansatz = PauliTwoDesign(2)

        # construct variational quantum regressor
        self.vqr = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=self.optimizers[optimizer_name],
            quantum_instance=self.backends[quantum_instance_name],
        )

    def setup_dataset_ccpp(self, X, y, quantum_instance_name, optimizer_name):
        """Training VQR for CCPP dataset."""

        scaler = MinMaxScaler((-1, 1))
        self.X = scaler.fit_transform(X)
        self.y = scaler.fit_transform(y.reshape(-1, 1))

        feature_map = ZFeatureMap(4)
        ansatz = PauliTwoDesign(4)

        self.vqr_fitted = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=self.optimizers[optimizer_name],
            quantum_instance=self.backends[quantum_instance_name],
        )

    def setup(self, dataset, quantum_instance_name, optimizer_name):
        """setup"""
        self.X = self.datasets[dataset]["features"]
        self.y = self.datasets[dataset]["labels"]

        if dataset == "dataset_synthetic_regression":
            self.setup_dataset_synthetic_regression(quantum_instance_name, optimizer_name)
        elif dataset == "dataset_ccpp":
            self.setup_dataset_ccpp(self.X, self.y, quantum_instance_name, optimizer_name)

    def time_fit_vqr(self, _, __, ___, ____):
        """Time fitting VQR to data."""

        self.vqr.fit(self.X, self.y)


if __name__ == "__main__":
    for dataset, backend, optimizer in product(*VqrFitBenchmarks.params):
        bench = VqrFitBenchmarks()
        try:
            bench.setup(dataset, backend, optimizer)
        except NotImplementedError:
            continue

        for method in ["time_fit_vqr"]:
            elapsed = timeit(
                f"bench.{method}(None, None, None, None)", number=10, globals=globals()
            )
            print(f"{method}:\t{elapsed}")
