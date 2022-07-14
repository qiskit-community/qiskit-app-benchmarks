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
import argparse
from typing import List
from timeit import timeit
from pathlib import Path
from qiskit_nature import settings
from qiskit_nature.hdf5 import save_to_hdf5
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver, HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

settings.dict_aux_operators = True

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class JordanWignerMapperBenchmarks:
    """Jordan-Wigner Mapper Benchmarks."""

    version = 1
    params: List[str] = [
        "H2 ParticleNumber",
        "H2 ElectronicEnergy",
        "H2 DipoleMomentX",
        "H2 DipoleMomentY",
        "H2 DipoleMomentZ",
        "H2 AngularMomentum",
        "H2 Magnetization",
        "H2O ParticleNumber",
        "H2O ElectronicEnergy",
        "H2O DipoleMomentX",
        "H2O DipoleMomentY",
        "H2O DipoleMomentZ",
        "H2O AngularMomentum",
        "H2O Magnetization",
        "LiH ParticleNumber",
        "LiH ElectronicEnergy",
        "LiH DipoleMomentX",
        "LiH DipoleMomentY",
        "LiH DipoleMomentZ",
        "LiH AngularMomentum",
        "LiH Magnetization",
    ]
    param_names = ["operator_type"]

    _hdf5_files = [
        ("H .0 .0 .0; H .0 .0 0.735", "jordan_wigner_benchmark_driver_H2.hdf5"),
        (
            "O 0.0 0.0 0.0; H 0.758602 0.0 0.504284; H 0.758602 0.0 -0.504284",
            "jordan_wigner_benchmark_driver_H2O.hdf5",
        ),
        ("Li 0.0 0.0 0.0; H 0.0 0.0 1.5474", "jordan_wigner_benchmark_driver_LiH.hdf5"),
    ]

    @staticmethod
    def make_hdf5_file():
        """create hdf5 files"""

        for _, (atom, file_name) in enumerate(JordanWignerMapperBenchmarks._hdf5_files):
            _driver = PySCFDriver(
                atom=atom,
                unit=UnitsType.ANGSTROM,
                basis="sto3g",
            )
            _molecule = _driver.run()
            save_to_hdf5(_molecule, file_name, replace=True)

    def setup_cache(self):
        """setup cache"""

        source_path = Path(__file__).resolve()
        source_dir = source_path.parent
        second_q_ops_list = []
        for _, file_name in JordanWignerMapperBenchmarks._hdf5_files:
            file_path = Path(source_dir, file_name)
            _hdf5_driver = HDF5Driver(file_path.resolve())
            _molecule = _hdf5_driver.run()
            atom_ops_list = list(_molecule.second_q_ops().values())
            for item in atom_ops_list:
                second_q_ops_list.append(item)

        return second_q_ops_list

    def setup(self, second_q_ops_list, operator):
        """setup"""
        self.op_number = self.params.index(operator)
        self.second_q_ops_list = second_q_ops_list
        self.jw_mapper = JordanWignerMapper()

    def time_map(self, _, __):
        """time map"""
        return self.jw_mapper.map(self.second_q_ops_list[self.op_number])


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Create hdf5 files")
    PARSER.add_argument(
        "-hdf5create", action="store_true", help="creates hdf5 file with PYSCFDriver results"
    )

    ARGS = PARSER.parse_args()
    if ARGS.hdf5create:
        JordanWignerMapperBenchmarks.make_hdf5_file()
    else:
        bench = JordanWignerMapperBenchmarks()
        second_q_ops_list = bench.setup_cache()
        for operator in JordanWignerMapperBenchmarks.params:
            bench = JordanWignerMapperBenchmarks()
            try:
                bench.setup(second_q_ops_list, operator)
            except NotImplementedError:
                continue

            for method in set(dir(JordanWignerMapperBenchmarks)):
                if method.startswith("time_"):
                    elapsed = timeit(
                        f"bench.{method}(None, '{operator}')",
                        number=10,
                        globals=globals(),
                    )
                    print(f"bench.{method} : \t{elapsed}")
