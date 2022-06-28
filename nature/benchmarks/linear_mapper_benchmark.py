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

import retworkx
from qiskit_nature.mappers.second_quantization import LinearMapper
from qiskit_nature.problems.second_quantization.lattice.models import IsingModel
from qiskit_nature.problems.second_quantization.lattice import Lattice

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class LinearMapperBenchmarks:
    """Linear Mapper Benchmarks."""

    version = 1
    params = [0]
    param_names = ["op_number"]

    def setup_cache(self):
        """setup cache"""
        graph = retworkx.PyGraph(multigraph=False)
        graph.add_nodes_from(list(range(20)))
        graph.add_edge(1, 5, 4)
        lattice = Lattice(graph)
        ising_model = IsingModel(lattice)
        second_q_ops = ising_model.second_q_ops()
        return second_q_ops

    def setup(self, second_q_ops, __):
        """setup"""
        self.second_q_ops = second_q_ops
        self.linear_mapper = LinearMapper()

    def time_map(self, _, __):
        """time map"""
        return self.linear_mapper.map(self.second_q_ops)


if __name__ == "__main__":
    bench = LinearMapperBenchmarks()
    second_q_ops = bench.setup_cache()
    for op_number in LinearMapperBenchmarks.params:
        bench = LinearMapperBenchmarks()
        try:
            bench.setup(second_q_ops, op_number)
        except NotImplementedError:
            continue

        for method in set(dir(LinearMapperBenchmarks)):
            if method.startswith("time_"):
                elapsed = timeit(f"bench.{method}(None, {op_number})", number=10, globals=globals())
                print(f"bench.{method} : \t{elapsed}")
