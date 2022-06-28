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

"""Jordan-Wigner Mapper Benchmarks."""
from typing import Union, List
from timeit import timeit
from itertools import product

from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class JordanWignerMapperBenchmarks:
    """Jordan-Wigner Mapper Benchmarks."""

    version = 1
    params: List[Union[List[int], List[str]]] = [list(range(3 * 7)), ["dense", "sparse"]]
    param_names = ["op_number", "display_format"]

    def setup_cache(self):
        """setup cache"""
        atom_list = [
            "H .0 .0 .0; H .0 .0 0.735",
            "O 0.0 0.0 0.0; H 0.758602 0.0 0.504284; H 0.758602 0.0 -0.504284",
            "Li 0.0 0.0 0.0; H 0.0 0.0 1.5474",
        ]
        second_q_ops_list = []
        for num_atom in range(3):
            _driver = PySCFDriver(atom=atom_list[num_atom], unit=UnitsType.ANGSTROM, basis="sto3g")
            _problem = ElectronicStructureProblem(_driver)
            second_q_ops_list.append(_problem.second_q_ops())
            print(len(second_q_ops_list))

        second_q_ops_list = [second_q_ops for atoms in second_q_ops_list for second_q_ops in atoms]
        return second_q_ops_list

    def setup(self, second_q_ops_list, _, __):
        """setup"""
        self.second_q_ops_list = second_q_ops_list
        self.jw_mapper = JordanWignerMapper()

    def time_map(self, _, op_number, __):
        """time map"""
        return self.jw_mapper.map(self.second_q_ops_list[op_number])


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
                elapsed = timeit(
                    f"bench.{method}(None, {op_number}, '{display_format}')",
                    number=10,
                    globals=globals(),
                )
                print(f"bench.{method} : \t{elapsed}")
