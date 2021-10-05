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

from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms import VQR

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_regressor_benchmark import BaseRegressorBenchmark


class VqrBenchmarks(BaseRegressorBenchmark):
    """Variational Quantum Regressor benchmarks."""

    version = 1
    timeout = 1200.0
    params = [["dataset_1"], ["qasm_simulator", "statevector_simulator"]]
    param_names = ["backend name"]

    def setup(self, dataset, quantum_instance_name):
        """setup"""
        self.X = self.datasets[dataset][:, 0].reshape(-1, 1)
        self.y = self.datasets[dataset][:, 1]

        # construct a feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        # construct an ansatz
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

        self.vqr_fitted.fit(self.X, self.y)

    def time_score_vqr(self, _, __):
        """Time scoring VQR on data."""

        self.vqr_fitted.score(self.X, self.y)

    def time_predict_vqr(self, _, __):
        """Time predicting with VQR."""

        y_predict = self.vqr_fitted.predict(self.X)
        return y_predict


if __name__ == "__main__":
    for dataset, backend in product(*VqrBenchmarks.params):
        bench = VqrBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue

        for method in ("time_score_vqr", "time_predict_vqr"):
            elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
            print(f"{method}:\t{elapsed}")
