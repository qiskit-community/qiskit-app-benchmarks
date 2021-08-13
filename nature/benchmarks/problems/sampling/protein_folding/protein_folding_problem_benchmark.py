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

"""Protein Folding Problem benchmarks."""

from timeit import timeit

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.problems.sampling.protein_folding import (
    PenaltyParameters,
    Peptide,
    ProteinFoldingProblem,
    RandomInteraction,
    MiyazawaJerniganInteraction,
    MixedInteraction,
)


# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class ProteinFoldingProblemBenchmarks:
    """Protein Folding Problem benchmarks."""

    peptide1 = ["APRLRFY", [""] * 7]  # Neuropeptide
    peptide2 = ["APRLRFY", ["", "", "R", "", "T", "W", ""]]  # Neuropeptide with dummy side chains
    peptide3 = ["DRVYIHPFHL", [""] * 10]  # Angiotensin I, human
    peptide4 = [
        "DRVYIHPFHL",
        ["", "", "P", "R", "L", "H", "Y", "", "I", ""],
    ]  # Angiotensin I, human with dummy side chains
    peptides = [peptide1, peptide2, peptide3, peptide4]
    interactions = [MiyazawaJerniganInteraction(), RandomInteraction(), MixedInteraction()]

    param_names = ["main chain residue", "side chain residue", "interaction type"]

    def setup(self, main_chain_residue_seq, side_chain_residue_sequences, interaction):
        """setup"""
        qasm_sim = Aer.get_backend("qasm_simulator")
        self._qins = QuantumInstance(backend=qasm_sim, shots=1)
        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        self.protein_folding_problem = ProteinFoldingProblem(
            peptide, interaction, PenaltyParameters()
        )

    def time_generate_full_qubit_operator(self, _, __):
        """Time generation of full protein folding qubit operator."""
        return self.protein_folding_problem._qubit_op_full()

    def time_generate_compressed_qubit_operator(self, _, __):
        """Time generation of compressed protein folding qubit operator."""
        return self.protein_folding_problem.qubit_op()


if __name__ == "__main__":
    for (
        main_chain_residue_seq,
        side_chain_residue_sequences,
    ) in ProteinFoldingProblemBenchmarks.peptides:
        for interaction in ProteinFoldingProblemBenchmarks.interactions:
            bench = ProteinFoldingProblemBenchmarks()
            try:
                bench.setup(main_chain_residue_seq, side_chain_residue_sequences, interaction)
            except NotImplementedError:
                continue
            for method in set(dir(ProteinFoldingProblemBenchmarks)):
                if method.startswith("time_"):
                    elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
                    print(
                        f"main_chain_residue_seq={main_chain_residue_seq}, "
                        f"side_chain_residue_sequences={side_chain_residue_sequences}, interaction="
                        f"{interaction} {method}:\t{elapsed}"
                    )
