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

"""Base for Classifier benchmarks."""

from abc import ABC

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals

DATASET_SYNTHETIC_CLASSIFICATION = "dataset_synthetic"
DATASET_IRIS_CLASSIFICATION = "dataset_iris"


class BaseClassifierBenchmark(ABC):
    """Base for Classifier benchmarks."""

    def __init__(self):
        algorithm_globals.random_seed = 12345
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
