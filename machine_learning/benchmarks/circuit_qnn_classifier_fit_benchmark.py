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
"""Circuit QNN Classifier benchmarks."""

from itertools import product
from timeit import timeit

from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD, L_BFGS_B

from .circuit_qnn_base_classifier_benchmark import CircuitQnnBaseClassifierBenchmark
from .base_classifier_benchmark import (
    DATASET_SYNTHETIC_CLASSIFICATION,
    DATASET_IRIS_CLASSIFICATION,
)


class CircuitQnnFitClassifierBenchmarks(CircuitQnnBaseClassifierBenchmark):
    """Circuit QNN Classifier benchmarks."""

    version = 2
    timeout = 1200.0
    params = (
        [DATASET_SYNTHETIC_CLASSIFICATION, DATASET_IRIS_CLASSIFICATION],
        ["qasm_simulator", "statevector_simulator"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
    )
    param_names = ["dataset", "backend name", "optimizer"]

    def __init__(self) -> None:
        super().__init__()

        self.optimizers = {
            "cobyla": COBYLA(maxiter=100),
            "nelder-mead": NELDER_MEAD(maxiter=50),
            "l-bfgs-b": L_BFGS_B(maxiter=20),
        }
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.model = None

    def setup(self, dataset: str, quantum_instance_name: str, optimizer: str) -> None:
        """Set up the benchmark."""
        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]

        if dataset == DATASET_SYNTHETIC_CLASSIFICATION:
            self.model = self._construct_qnn_classifier_synthetic(
                quantum_instance_name=quantum_instance_name,
                optimizer=self.optimizers[optimizer],
            )
        elif dataset == DATASET_IRIS_CLASSIFICATION:
            self.model = self._construct_qnn_classifier_iris(
                quantum_instance_name=quantum_instance_name,
                optimizer=self.optimizers[optimizer],
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    # pylint: disable=invalid-name
    def time_fit_circuit_qnn_classifier(self, _, __, ___):
        """Time fitting CircuitQNN classifier to data."""
        self.model.fit(self.train_features, self.train_labels)


if __name__ == "__main__":
    for dataset_name, backend, optimizer_name in product(*CircuitQnnFitClassifierBenchmarks.params):
        bench = CircuitQnnFitClassifierBenchmarks()
        try:
            bench.setup(dataset_name, backend, optimizer_name)
        except NotImplementedError:
            continue

        for method in ["time_fit_circuit_qnn_classifier"]:
            elapsed = timeit(
                f'bench.{method}("{dataset_name}", "{backend}", "{optimizer_name}")',
                number=10,
                globals=globals(),
            )
            print(f"{method}:\t{elapsed}")
