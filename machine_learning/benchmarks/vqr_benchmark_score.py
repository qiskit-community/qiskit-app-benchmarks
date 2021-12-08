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

from joblib import dump, load
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliTwoDesign, ZFeatureMap
from qiskit_machine_learning.algorithms import VQR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_regressor_benchmark import BaseRegressorBenchmark


class VqrScoreBenchmarks(BaseRegressorBenchmark):
    """Variational Quantum Regressor benchmarks."""

    version = 1
    timeout = 1200.0
    params = [
        ["dataset_synthetic_regression", "dataset_ccpp"],
        ["qasm_simulator", "statevector_simulator"],
    ]
    param_names = ["dataset", "backend name"]

    def setup_dataset_synthetic_regression(self, X, y, quantum_instance_name):
        """Training VQR function for synthetic regression dataset."""

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        # construct simple ansatz
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)

        # construct variational quantum regressor
        self.vqr_fitted = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=L_BFGS_B(),
            quantum_instance=self.backends[quantum_instance_name],
        )

        try:
            self.vqr_fitted._fit_result = load(
                f"/tmp/dataset_synthetic_regression_{quantum_instance_name}.obj"
            )
        except FileNotFoundError:
            # fit regressor
            self.vqr_fitted.fit(self.X_train, self.y_train)

        self.pred = self.vqr_fitted.predict(self.X_test)

    def setup_dataset_ccpp(self, X, y, quantum_instance_name):
        """Training VQR for CCPP dataset."""

        scaler = MinMaxScaler((-1, 1))
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y.reshape(-1, 1))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        feature_map = ZFeatureMap(4)
        ansatz = PauliTwoDesign(4)

        self.vqr_fitted = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=L_BFGS_B(),
            quantum_instance=self.backends[quantum_instance_name],
        )

        try:
            self.vqr_fitted._fit_result = load(f"/tmp/dataset_ccpp_{quantum_instance_name}.obj")
        except FileNotFoundError:
            # fit regressor
            self.vqr_fitted.fit(self.X_train, self.y_train)

        self.pred = self.vqr_fitted.predict(self.X_test)

    def setup(self, dataset, quantum_instance_name):
        """setup"""

        self.X = self.datasets[dataset]["features"]
        self.y = self.datasets[dataset]["labels"]

        if dataset == "dataset_synthetic_regression":
            self.setup_dataset_synthetic_regression(self.X, self.y, quantum_instance_name)
        elif dataset == "dataset_ccpp":
            self.setup_dataset_ccpp(self.X, self.y, quantum_instance_name)

    def setup_cache(self):
        """Cache VQR fitted model"""
        for dataset, backend in product(*self.params):
            self.setup(dataset, backend)

            dump(self.vqr_fitted._fit_result, f"/tmp/{dataset}_{backend}.obj")

    def track_r2_score(self, _, __):
        """R2 score of VQR on data."""

        r2score = r2_score(y_true=self.y_test, y_pred=self.pred)
        return r2score

    def track_mae(self, _, __):
        """Mean absolute error of VQR on data."""

        mae = mean_absolute_error(y_true=self.y_test, y_pred=self.pred)
        return mae

    def track_mse(self, _, __):
        """Mean squared error of VQR on data."""

        mse = mean_squared_error(y_true=self.y_test, y_pred=self.pred)
        return mse


if __name__ == "__main__":
    for dataset, backend in product(*VqrScoreBenchmarks.params):
        bench = VqrScoreBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue
