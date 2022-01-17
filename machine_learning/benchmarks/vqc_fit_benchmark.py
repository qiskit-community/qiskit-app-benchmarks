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
"""Variational Quantum Classifier benchmarks."""

from itertools import product
from timeit import timeit

from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD

from .base_classifier_benchmark import DATASET_SYNTHETIC_CLASSIFICATION
from .vqc_base_benchmark import VqcBaseClassifierBenchmark


class VqcFitBenchmarks(VqcBaseClassifierBenchmark):
    """Variational Quantum Classifier benchmarks."""

    version = 2
    timeout = 1200.0
    params = (
        # VQC does not work with multiple classes, so only the synthetic dataset now
        [DATASET_SYNTHETIC_CLASSIFICATION],
        ["qasm_simulator", "statevector_simulator"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
        ["cross_entropy", "squared_error"],
    )
    param_names = ["dataset", "backend name", "optimizer", "loss function"]

    def __init__(self):
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

    def setup(self, dataset: str, quantum_instance_name: str, optimizer: str, loss_function: str):
        """Setup the benchmark."""
        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]

        if dataset == DATASET_SYNTHETIC_CLASSIFICATION:
            self.model = self._construct_vqc_classifier_synthetic(
                quantum_instance_name=quantum_instance_name,
                optimizer=self.optimizers[optimizer],
                loss_function=loss_function,
            )
        else:
            self.model = self._construct_vqc_classifier_iris(
                quantum_instance_name=quantum_instance_name,
                optimizer=self.optimizers[optimizer],
                loss_function=loss_function,
            )

    # pylint: disable=invalid-name
    def time_fit_vqc(self, _, __, ___, ____):
        """Time fitting VQC to data."""
        self.model.fit(self.train_features, self.train_labels)


if __name__ == "__main__":
    for dataset_name, backend_name, optimizer_name, loss_function_name in product(
        *VqcFitBenchmarks.params
    ):
        bench = VqcFitBenchmarks()
        try:
            bench.setup(dataset_name, backend_name, optimizer_name, loss_function_name)
        except NotImplementedError:
            continue

        for method in ["time_fit_vqc"]:
            elapsed = timeit(
                f'bench.{method}("{dataset_name}", "{backend_name}", '
                f'"{optimizer_name}", "{loss_function_name}")',
                number=10,
                globals=globals(),
            )
            print(f"{method}:\t{elapsed}")
