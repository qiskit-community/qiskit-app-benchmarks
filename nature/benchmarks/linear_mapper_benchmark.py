# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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
import rustworkx
from qiskit_nature.second_q.mappers import LinearMapper
from qiskit_nature.second_q.hamiltonians import IsingModel
from qiskit_nature.second_q.hamiltonians.lattices import Lattice
from qiskit_nature.settings import settings

settings.use_pauli_sum_op = False

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class LinearMapperBenchmarks:
    """Linear Mapper Benchmarks."""

    version = 2
    timeout = 120.0
    params = [50, 80, 100]
    param_names = ["Number of nodes"]

    def setup_cache(self):
        """setup cache"""

        second_q_ops_list = []
        edge_arr = list((n**11) % 100 for n in range(100))

        for index, _ in enumerate(self.params):
            graph = rustworkx.PyGraph(multigraph=False)
            graph.add_nodes_from(list(range(self.params[index])))

            for i in range(self.params[index]):
                for j in range(i + 1, self.params[index]):
                    graph.add_edge(i, j, edge_arr[i])

            lattice = Lattice(graph)
            ising_model = IsingModel(lattice)
            second_q_op = ising_model.second_q_op()
            second_q_ops_list.append(second_q_op)

        return second_q_ops_list

    def setup(self, second_q_ops_list, num_nodes):
        """setup"""
        self.second_q_ops_list = second_q_ops_list
        self.op_number = self.params.index(num_nodes)
        self.linear_mapper = LinearMapper()

    def time_map(self, _, __):
        """time map"""
        return self.linear_mapper.map(self.second_q_ops_list[self.op_number])


if __name__ == "__main__":
    bench = LinearMapperBenchmarks()
    second_q_ops_list = bench.setup_cache()
    for num_nodes in LinearMapperBenchmarks.params:
        bench = LinearMapperBenchmarks()
        try:
            bench.setup(second_q_ops_list, num_nodes)
        except NotImplementedError:
            continue

        for method in set(dir(LinearMapperBenchmarks)):
            if method.startswith("time_"):
                elapsed = timeit(f"bench.{method}(None, {num_nodes})", number=10, globals=globals())
                print(f"bench.{method} : \t{elapsed}")
