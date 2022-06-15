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

"""Jordan-Wigner Mapper Benchmarks."""
from timeit import timeit
from itertools import product

import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance

from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init

class JordanWignerMapperBenchmarks:
    """Jordan Wigner Mapper Benchamrks"""

    version = 1
    params = [[i for i in range(7)], ["dense", "sparse"]]
    param_names = ["op_number", "display_format"]

    def setup_cache(self):
        _driver = PySCFDriver(atom = "O 0.0 0.0 0.0; H 0.758602 0.0 0.504284; H 0.758602 0.0 -0.504284",
                              unit=UnitsType.ANGSTROM,
                              basis='sto3g')
        _problem = ElectronicStructureProblem(_driver)
        second_q_ops_list = _problem.second_q_ops()
        return second_q_ops_list

    def setup(self, second_q_ops_list, op_number, display_format):
        self.second_q_ops_list = second_q_ops_list
        self.jw_mapper = JordanWignerMapper()
        self.parity_mapper = ParityMapper()

    def time_map(self, _, op_number, __):
        return self.parity_mapper.map(self.second_q_ops_list[op_number])
        # return self.jw_mapper.map(self.second_q_ops_list[op_number])


if __name__ == "__main__":
    bench = JordanWignerMapperBenchmarks()
    second_q_ops_list = bench.setup_cache()
    for op_number, display_format in product(*JordanWignerMapperBenchmarks.params):
        bench = JordanWignerMapperBenchmarks()
        try:
            bench.setup(second_q_ops_list, op_number, display_format)
        except NotImplementedError:
            continue

    for method in set(dir(JordanWignerMapperBenchmarks)):
        if method.startswith("time_"):
            elapsed = timeit(f"bench.{method}", number = 10, globals = globals())
            print(f"bench.{method} : \t{elapsed}")
