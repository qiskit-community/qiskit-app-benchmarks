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
"""Base for Regressor benchmarks."""

from abc import ABC
from qiskit import Aer
from qiskit.utils import QuantumInstance
from .datasets import (
    DATASET_SYNTHETIC_REGRESSION_FEATURES,
    DATASET_SYNTHETIC_REGRESSION_LABELS,
    DATASET_CCPP_FEATURES,
    DATASET_CCPP_LABELS,
)


class BaseRegressorBenchmark(ABC):
    """Base for Regressor benchmarks."""

    def __init__(self):

        quantum_instance_statevector = QuantumInstance(
            Aer.get_backend("statevector_simulator"), shots=1024
        )
        quantum_instance_qasm = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=1024)

        self.backends = {
            "statevector_simulator": quantum_instance_statevector,
            "qasm_simulator": quantum_instance_qasm,
        }

        self.datasets = {
            "dataset_synthetic_regression": {
                "features": DATASET_SYNTHETIC_REGRESSION_FEATURES,
                "labels": DATASET_SYNTHETIC_REGRESSION_LABELS,
            },
            "dataset_ccpp": {"features": DATASET_CCPP_FEATURES, "labels": DATASET_CCPP_LABELS},
        }
