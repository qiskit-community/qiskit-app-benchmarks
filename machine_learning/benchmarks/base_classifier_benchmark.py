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

        self.dataset_1 = np.array(
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

        self.dataset_iris_features = np.array(
            [
                [4.7, 3.2, 1.3, 0.2],
                [4.9, 3.1, 1.5, 0.1],
                [5.8, 4.0, 1.2, 0.2],
                [5.0, 3.0, 1.6, 0.2],
                [4.8, 3.1, 1.6, 0.2],
                [5.2, 4.1, 1.5, 0.1],
                [4.4, 3.0, 1.3, 0.2],
                [5.0, 3.5, 1.3, 0.3],
                [4.6, 3.2, 1.4, 0.2],
                [5.3, 3.7, 1.5, 0.2],
                [7.0, 3.2, 4.7, 1.4],
                [5.5, 2.3, 4.0, 1.3],
                [5.9, 3.0, 4.2, 1.5],
                [5.9, 3.2, 4.8, 1.8],
                [6.6, 3.0, 4.4, 1.4],
                [5.5, 2.4, 3.7, 1.0],
                [6.7, 3.1, 4.7, 1.5],
                [6.1, 3.0, 4.6, 1.4],
                [5.7, 2.9, 4.2, 1.3],
                [5.1, 2.5, 3.0, 1.1],
                [7.1, 3.0, 5.9, 2.1],
                [6.5, 3.2, 5.1, 2.0],
                [6.0, 2.2, 5.0, 1.5],
                [6.1, 3.0, 4.9, 1.8],
                [7.7, 3.0, 6.1, 2.3],
                [6.0, 3.0, 4.8, 1.8],
                [5.8, 2.7, 5.1, 1.9],
                [6.7, 3.0, 5.2, 2.3],
                [6.2, 3.4, 5.4, 2.3],
                [5.9, 3.0, 5.1, 1.8],
            ]
        )

        self.dataset_iris_labels = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        )

        self.datasets = {"dataset_1": self.dataset_1, "dataset_iris": {"features": self.dataset_iris_features, "labels": self.dataset_iris_labels}}
