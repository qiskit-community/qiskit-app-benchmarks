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

"""Linear Mapper Benchmarks."""
from timeit import timeit
import numpy as np

import retworkx
from qiskit_nature.mappers.second_quantization import LinearMapper
from qiskit_nature.problems.second_quantization.lattice.models import IsingModel
from qiskit_nature.problems.second_quantization.lattice import Lattice

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class LinearMapperBenchmarks:
    """Linear Mapper Benchmarks."""

    version = 1
    seed = 100
    params = list(range(3))
    param_names = ["op_number"]

    def setup_cache(self):
        """setup cache"""

        second_q_ops_list = []

        for _ in range(3):
            num_nodes = np.random.randint(40)
            graph = retworkx.PyGraph(multigraph=False)
            graph.add_nodes_from(list(range(num_nodes)))

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if j != i:
                        graph.add_edge(i, j, np.random.randint(100))

            lattice = Lattice(graph)
            ising_model = IsingModel(lattice)
            second_q_ops = ising_model.second_q_ops()
            second_q_ops_list.append(second_q_ops)

        return second_q_ops_list

    def setup(self, second_q_ops_list, __):
        """setup"""
        self.second_q_ops_list = second_q_ops_list
        self.linear_mapper = LinearMapper()

    def time_map(self, _, op_number):
        """time map"""
        return self.linear_mapper.map(self.second_q_ops_list[op_number])


if __name__ == "__main__":
    bench = LinearMapperBenchmarks()
    second_q_ops_list = bench.setup_cache()
    for op_number in LinearMapperBenchmarks.params:
        bench = LinearMapperBenchmarks()
        try:
            bench.setup(second_q_ops_list, op_number)
        except NotImplementedError:
            continue

        for method in set(dir(LinearMapperBenchmarks)):
            if method.startswith("time_"):
                elapsed = timeit(f"bench.{method}(None, {op_number})", number=10, globals=globals())
                print(f"bench.{method} : \t{elapsed}")
