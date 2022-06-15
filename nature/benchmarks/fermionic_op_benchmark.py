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

"""Fermionic Operator Benchmarks."""
from timeit import timeit
from itertools import product

from qiskit_nature.operators.second_quantization import FermionicOp

class FermionicOperatorBenchmarks:
    """Fermionic Operator Benchamrks"""

    seed = 100
    version = 1
    params = [[i for i in range(10)], ["dense", "sparse"]] # 3125
    param_names = ["op_number", "display_format"]

    # params = [["I", "+", "-", "N", "E"], [np.random.randint(20)], ["sparse", "dense"]]
    # param_names = ["label", "register_length", "display_format"]

    def setup_cache(self):
        label_list = ["".join(label) for label in product(["I", "+", "-", "N", "E"], repeat=5)]  # Total number of labels produced is 3125.
        return label_list

    def setup(self, label_list, op_number, display_format):
        self.label_list = label_list

    def time_FermionicOp(self, _, op_number, display_format):
        return FermionicOp(self.label_list[op_number], display_format)

if __name__ == "__main__":
    bench = FermionicOperatorBenchmarks()
    label_list = bench.setup_cache()
    for op_number, display_format in product(*FermionicOperatorBenchmarks.params):
        bench = FermionicOperatorBenchmarks()
        try:
            bench.setup(label_list, op_number, display_format)
        except NotImplementedError:
            continue

    for method in set(dir(FermionicOperatorBenchmarks)):
        if method.startswith("time_"):
            elapsed = timeit(f"bench.{method}", number = 10, globals = globals())
            print(f"bench.{method} : \t{elapsed}")



# List of functins in FermionicOp class.
    # def time_add(self): # arg: other FermionicOp
    # def time_adjoint(self): # no arg
    # def time_compose(self): # arg: other FermionicOp
    # def time_mul(self): # arg: other complex (scalar maybe)
    # def time_one(self): # arg: int register_len
    # def time_reduce(self): #DEPRECATION; instead use simplify or normal_ordered
    # def time_set_truncation(self): # class method with cls as argument
    # def time_to_list(self): #arg: display_format (optional)
    # def time_to_matrix(self):
    # def time_to_normal_order(self): #DEPRECATION; instead use normal_ordered
    # def time_zero(self): # arg: int register_len
    # def time_simplify(self):
    # def time_normal_ordered(self): # no arg
