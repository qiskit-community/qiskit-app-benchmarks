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

"""Base for Classifier benchmarks."""

from abc import ABC
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance


class BaseClassifierBenchmark(ABC):
    """Base for Classifier benchmarks."""

    def __init__(self):

        quantum_instance_statevector = QuantumInstance(
            Aer.get_backend("statevector_simulator"), shots=1024
        )
        quantum_instance_qasm = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=1024)

        self.backends = {
            "statevector_simulator": quantum_instance_statevector,
            "qasm_simulator": quantum_instance_qasm,
        }

        self.dataset_synthetic = np.array(
            [
                [0.63332707, 0.05700334],
                [-0.04218316, -0.74066734],
                [-0.54731074, 0.6997243],
                [-0.03254765, 0.68657814],
                [0.57025591, 0.67333428],
                [0.32978679, 0.90721741],
                [0.28112104, -0.52329682],
                [0.03209235, 0.05112333],
                [0.46215367, 0.97636782],
                [0.93945321, 0.09375981],
                [-0.6546925, 0.1612654],
                [0.38871208, 0.73535322],
                [-0.72805702, 0.73124097],
                [-0.79972062, -0.84444756],
                [0.87636701, -0.66912929],
                [-0.08563266, 0.79913683],
                [0.31805884, -0.84938654],
                [0.96364301, 0.86688318],
                [-0.50482284, -0.64370197],
                [-0.41502205, 0.38414452],
            ]
        )
        self.datasets = {"dataset_synthetic": self.dataset_synthetic}
