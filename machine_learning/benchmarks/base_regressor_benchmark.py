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
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance


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

        self.dataset_1 = np.array(
            [
                [-0.85774348, -0.74783684],
                [2.26973107, 0.82203840],
                [0.28728974, 0.16437351],
                [-1.49685228, -1.06793086],
                [-0.12013558, -0.12098070],
                [-1.10343779, -0.95175504],
                [1.42191594, 1.18888902],
                [1.95396369, 0.94571940],
                [-2.08711011, -0.71775022],
                [1.84776896, 0.83823119],
                [-2.31663345, -0.62089123],
                [0.41193372, 0.30567915],
                [-1.59525663, -1.13101328],
                [2.61017730, 0.47487221],
                [-3.02161441, -0.20629148],
                [-0.79122301, -0.89714515],
                [2.68199945, 0.41766852],
                [-2.88543292, -0.18996203],
                [-0.05563549, 0.12252040],
                [-1.79495492, -1.07527775],
            ],
        )
        self.datasets = {"dataset_1": self.dataset_1}
