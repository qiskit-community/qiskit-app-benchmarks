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

from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD

from .base_regressor_benchmark import (
    BaseRegressorBenchmark,
    DATASET_SYNTHETIC_REGRESSION,
    DATASET_CCPP_REGRESSION,
)


class OpflowQnnFitRegressorBenchmarks(BaseRegressorBenchmark):
    """OpflowQNN Regressor benchmarks."""

    version = 1
    timeout = 1200.0
    params = (
        [DATASET_SYNTHETIC_REGRESSION, DATASET_CCPP_REGRESSION],
        ["qasm_simulator", "statevector_simulator"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
    )
    param_names = ["dataset", "backend name", "optimizer"]

    def __init__(self):
        super().__init__()
        self.optimizers = {
            "cobyla": COBYLA(maxiter=100),
            "nelder-mead": NELDER_MEAD(maxiter=50),
            "l-bfgs-b": L_BFGS_B(maxiter=20),
        }
        self.train_features = None
        self.train_labels = None
        self.model = None

    def setup(self, dataset: str, quantum_instance_name: str, optimizer: str):
        """Setup the benchmark."""
        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]

        if dataset == DATASET_SYNTHETIC_REGRESSION:
            self.model = self._construct_qnn_synthetic(
                quantum_instance_name, self.optimizers[optimizer]
            )
        elif dataset == DATASET_CCPP_REGRESSION:
            self.model = self._construct_qnn_ccpp(quantum_instance_name, self.optimizers[optimizer])

    # pylint: disable=invalid-name
    def time_fit_opflow_qnn_regressor(self, _, __, ___):
        """Time fitting OpflowQNN regressor to data."""
        self.model.fit(self.train_features, self.train_labels)


if __name__ == "__main__":
    for dataset_name, backend, optimizer_name in product(*OpflowQnnFitRegressorBenchmarks.params):
        bench = OpflowQnnFitRegressorBenchmarks()
        try:
            bench.setup(dataset_name, backend, optimizer_name)
        except NotImplementedError:
            continue

        for method in ["time_fit_opflow_qnn_regressor"]:
            elapsed = timeit(
                f'bench.{method}("{dataset_name}", "{backend}", "{optimizer_name}")',
                number=10,
                globals=globals(),
            )
            print(f"{method}:\t{elapsed}")
