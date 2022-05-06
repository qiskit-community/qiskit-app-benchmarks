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
"""PegasosQSVC benchmarks."""

from itertools import product
from timeit import timeit
from typing import Optional, Union

import numpy as np
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD

from .base_classifier_benchmark import DATASET_SYNTHETIC_CLASSIFICATION, DATASET_IRIS_CLASSIFICATION
from qiskit_machine_learning.kernels import QuantumKernel

from qiskit_machine_learning.algorithms import PegasosQSVC

from .pegasosQsvc_base_benchmark import PegasosQsvcBaseClassifierBenchmark


class PegasosQsvcFitBenchmarks(PegasosQsvcBaseClassifierBenchmark):
    """PegasosQSVC fit benchmarks."""

    version = 2
    timeout = 1200.0
    params = (
        # Only the synthetic dataset now
        [DATASET_SYNTHETIC_CLASSIFICATION],
        ["qasm_simulator", "statevector_simulator"],
        ["QuantumKernel"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
        ["cross_entropy", "squared_error"],
    )
    param_names = ["dataset", "backend name", "optimizer", "loss function"]

    def __init__(self) -> None:
        super().__init__()

        self.optimizers = {
            "cobyla": COBYLA(maxiter=100),
            "nelder-mead": NELDER_MEAD(maxiter=50),
            "l-bfgs-b": L_BFGS_B(maxiter=20),
        }
        self.train_features: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.model: Optional[QuantumKernel] = None

    def setup(
        self, 
        dataset: str, 
        quantum_instance_name: str, 
        optimizer: str, 
        loss_function: str
    ) -> None:
        """Set up the benchmark."""
        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]
        #new
        n_qubits = self.train_features.shape[1]
        if  dataset == DATASET_SYNTHETIC_CLASSIFICATION:
            _kernel = self._construct_QuantumKernel_classical_classifier(quantum_instance_name= quantum_instance_name, optimizer = optimizer,
                                                            loss_function = loss_function, num_qubits = n_qubits) #this is just a kernel matrix
        elif dataset == DATASET_IRIS_CLASSIFICATION:
            _kernel = self._construct_QuantumKernelTrainer(quantum_instance_name= quantum_instance_name, optimizer= optimizer, 
                                                            loss_function = loss_function, num_qubits = n_qubits, 
                ) #this is a classifier
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")


    # pylint: disable=invalid-name
    def time_fit_pegasosQsvc(self, _, __, ___, ____):
        """Time fitting QSVC to data."""
        self.model = PegasosQSVC(kernel = _kernel.evaluate)
        self.model.fit(self.train_features, self.train_labels)


if __name__ == "__main__":
    for dataset_name, backend_name, optimizer_name, loss_function_name in product(
        *PegasosQsvcFitBenchmarks.params
    ):
        bench = PegasosQsvcFitBenchmarks()
        try:
            bench.setup(dataset_name, backend_name, optimizer_name, loss_function_name)
        except NotImplementedError:
            continue
        for method in ["time_fit_pegasosQsvc"]:
            elapsed = timeit(
                f'bench.{method}("{dataset_name}", "{backend_name}", '
                f'"{optimizer_name}", "{loss_function_name}")',
                number=10,
                globals=globals(),
            )
            print(f"{method}:\t{elapsed}")
